import streamlit as st
import os
from pathlib import Path
import shutil
from PIL import Image, ImageDraw, ImageFont
import yaml
import json
from datetime import datetime
import csv
import streamlit.components.v1 as components

# Configure page - must be first Streamlit command
st.set_page_config(
    page_title="YOLO Reviewer",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Production configs
CACHE_SIZE = 50  # Number of images to keep in memory

# Image size presets - Updated Tiny to 240x240
IMAGE_SIZE_PRESETS = {
    'Tiny (Fastest)': (240, 240),        # 240x240 - ultra tiny
    'Compact (Fast)': (640, 360),        # 360p - Fast (NEW DEFAULT)
    'Small (Balanced)': (800, 450),      # 450p - Balanced
    'Medium (Quality)': (960, 540),      # 540p - Good quality
    'Large (Max)': (1280, 720)           # 720p - Maximum
}

class ReviewLogger:
    """Handles logging of review decisions"""
    
    def __init__(self, root_folder):
        self.log_file = os.path.join(root_folder, 'review_log.csv')
        self._ensure_log_exists()
    
    def _ensure_log_exists(self):
        """Create log file with headers if it doesn't exist"""
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'image_path', 'decision', 'previous_decision', 'session_id'])
    
    def log_decision(self, image_path, decision, previous_decision=None, session_id=None):
        """Log a review decision"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, image_path, decision, previous_decision or 'none', session_id or 'unknown'])
    
    def get_stats(self):
        """Get statistics from log file"""
        if not os.path.exists(self.log_file):
            return {'total': 0, 'accepted': 0, 'rejected': 0}
        
        stats = {'total': 0, 'accepted': 0, 'rejected': 0}
        try:
            with open(self.log_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    stats['total'] += 1
                    if row['decision'] == 'accepted':
                        stats['accepted'] += 1
                    elif row['decision'] == 'rejected':
                        stats['rejected'] += 1
        except:
            pass
        return stats

@st.cache_data(max_entries=CACHE_SIZE)
def load_image(image_path, max_size=(1280, 720)):
    """Load and cache image with size optimization"""
    try:
        img = Image.open(image_path)
        # Resize if too large
        if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
        return img
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None

@st.cache_data
def load_class_names(root_folder):
    """Load class names from data.yaml"""
    yaml_path = os.path.join(root_folder, 'data.yaml')
    if not os.path.exists(yaml_path):
        return {}
    
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            if 'names' in data:
                if isinstance(data['names'], list):
                    return {i: name for i, name in enumerate(data['names'])}
                elif isinstance(data['names'], dict):
                    return data['names']
    except:
        pass
    return {}

def parse_yolo_annotation(txt_path, img_width, img_height):
    """Parse YOLO format annotation"""
    boxes = []
    if not os.path.exists(txt_path):
        return boxes
    
    try:
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])
                    
                    # Convert to pixel coordinates
                    x_center_px = x_center * img_width
                    y_center_px = y_center * img_height
                    width_px = width * img_width
                    height_px = height * img_height
                    
                    x1 = int(x_center_px - width_px / 2)
                    y1 = int(y_center_px - height_px / 2)
                    x2 = int(x_center_px + width_px / 2)
                    y2 = int(y_center_px + height_px / 2)
                    
                    boxes.append({'class_id': class_id, 'bbox': (x1, y1, x2, y2)})
    except:
        pass
    return boxes

def draw_boxes(image, boxes, class_names=None):
    """Draw bounding boxes on image"""
    draw = ImageDraw.Draw(image)
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'cyan']
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    for box in boxes:
        class_id = box['class_id']
        bbox = box['bbox']
        color = colors[class_id % len(colors)]
        
        draw.rectangle(bbox, outline=color, width=2)
        
        label = class_names.get(class_id, f"C{class_id}") if class_names else f"C{class_id}"
        draw.text((bbox[0], max(0, bbox[1] - 18)), label, fill=color, font=font)
    
    return image

@st.cache_data
def find_image_pairs(root_folder):
    """Find all image-label pairs"""
    pairs = []
    root_path = Path(root_folder)
    
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    for ext in extensions:
        for img_path in root_path.rglob(ext):
            # Skip accepted/rejected folders
            if 'accepted' in str(img_path) or 'rejected' in str(img_path):
                continue
            
            # Find label
            label_path = Path(str(img_path).replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt')
            if label_path.exists():
                pairs.append({'image': str(img_path), 'label': str(label_path)})
    
    return pairs

def copy_files(image_path, label_path, dest_folder):
    """Copy files to destination"""
    os.makedirs(os.path.join(dest_folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(dest_folder, 'labels'), exist_ok=True)
    
    shutil.copy2(image_path, os.path.join(dest_folder, 'images', os.path.basename(image_path)))
    shutil.copy2(label_path, os.path.join(dest_folder, 'labels', os.path.basename(label_path)))

def remove_files(image_path, label_path, folder_type, root_folder):
    """Remove files from folder"""
    folder_path = os.path.join(root_folder, folder_type)
    for fname in [os.path.basename(image_path), os.path.basename(label_path)]:
        for subdir in ['images', 'labels']:
            fpath = os.path.join(folder_path, subdir, fname)
            if os.path.exists(fpath):
                os.remove(fpath)

def get_processed_images(root_folder):
    """Get list of already processed images from accepted/rejected folders"""
    processed = set()
    
    for folder in ['accepted', 'rejected']:
        folder_path = os.path.join(root_folder, folder, 'images')
        if os.path.exists(folder_path):
            for img_file in os.listdir(folder_path):
                processed.add(img_file)
    
    return processed

def find_first_unprocessed(pairs, processed_images):
    """Find index of first unprocessed image"""
    for idx, pair in enumerate(pairs):
        img_filename = os.path.basename(pair['image'])
        if img_filename not in processed_images:
            return idx
    return 0

def handle_decision(decision):
    """Handle accept/reject decision"""
    if len(st.session_state.pairs) == 0:
        return
    
    idx = st.session_state.current_index
    pair = st.session_state.pairs[idx]
    
    # Get previous decision
    prev_decision = st.session_state.decisions.get(idx)
    
    # Remove from old location if decision changed
    if prev_decision and prev_decision != decision:
        remove_files(pair['image'], pair['label'], prev_decision, st.session_state.root_folder)
        if prev_decision == 'accepted':
            st.session_state.accepted_count -= 1
        else:
            st.session_state.rejected_count -= 1
    
    # Copy to new location
    dest = os.path.join(st.session_state.root_folder, decision)
    copy_files(pair['image'], pair['label'], dest)
    
    # Update counts
    if not prev_decision or prev_decision != decision:
        if decision == 'accepted':
            st.session_state.accepted_count += 1
        else:
            st.session_state.rejected_count += 1
    
    # Log decision
    st.session_state.logger.log_decision(
        pair['image'], 
        decision, 
        prev_decision,
        st.session_state.session_id
    )
    
    # Update decision tracking
    st.session_state.decisions[idx] = decision
    
    # Auto-advance to next image
    if idx < len(st.session_state.pairs) - 1:
        st.session_state.current_index += 1

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.current_index = 0
    st.session_state.pairs = []
    st.session_state.root_folder = ""
    st.session_state.accepted_count = 0
    st.session_state.rejected_count = 0
    st.session_state.decisions = {}
    st.session_state.class_names = {}
    st.session_state.logger = None
    st.session_state.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    st.session_state.image_size_preset = 'Compact (Fast)'
    st.session_state.processed_count = 0
    st.session_state.initialized = True

# Sidebar
with st.sidebar:
    st.title("⚙️ Configuration")
    
    root_folder = st.text_input(
        "Dataset Path",
        value=st.session_state.root_folder,
        placeholder="/path/to/dataset/"
    )
    
    # Image size selector
    st.session_state.image_size_preset = st.selectbox(
        "Image Size",
        options=list(IMAGE_SIZE_PRESETS.keys()),
        index=list(IMAGE_SIZE_PRESETS.keys()).index(st.session_state.image_size_preset),
        help="Smaller = faster review speed. Compact is recommended for most use."
    )
    
    # Show current resolution
    current_size = IMAGE_SIZE_PRESETS[st.session_state.image_size_preset]
    st.caption(f"📐 Resolution: {current_size[0]}×{current_size[1]}px")
    
    if st.button("🔍 Load Dataset", type="primary", use_container_width=True):
        if os.path.exists(root_folder):
            with st.spinner("Loading dataset..."):
                st.session_state.root_folder = root_folder
                st.session_state.pairs = find_image_pairs(root_folder)
                st.session_state.class_names = load_class_names(root_folder)
                st.session_state.decisions = {}
                st.session_state.accepted_count = 0
                st.session_state.rejected_count = 0
                st.session_state.logger = ReviewLogger(root_folder)
                st.session_state.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                # Check for already processed images
                processed_images = get_processed_images(root_folder)
                st.session_state.processed_count = len(processed_images)
                
                # Find first unprocessed image
                first_unprocessed = find_first_unprocessed(st.session_state.pairs, processed_images)
                st.session_state.current_index = first_unprocessed
                
                st.success(f"✅ Loaded {len(st.session_state.pairs)} pairs")
                if st.session_state.class_names:
                    st.info(f"📝 {len(st.session_state.class_names)} classes loaded")
                
                if st.session_state.processed_count > 0:
                    st.info(f"📋 Already processed: {st.session_state.processed_count} | Starting at image {first_unprocessed + 1}")
        else:
            st.error("❌ Folder not found")
    
    st.divider()
    
    # Statistics
    if st.session_state.pairs:
        st.subheader("📊 Stats")
        
        total = len(st.session_state.pairs)
        reviewed = len(st.session_state.decisions)
        progress = reviewed / total if total > 0 else 0
        
        # Progress bar
        st.progress(progress, text=f"{reviewed}/{total} ({progress*100:.1f}%)")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("✅ Accept", st.session_state.accepted_count)
            st.metric("📋 Done", st.session_state.processed_count)
        with col2:
            st.metric("❌ Reject", st.session_state.rejected_count)
            remaining = total - reviewed - st.session_state.processed_count
            st.metric("⏳ Left", remaining)
        
        # Current status
        if st.session_state.current_index in st.session_state.decisions:
            status = st.session_state.decisions[st.session_state.current_index]
            if status == 'accepted':
                st.success("✅ Current: ACCEPTED")
            else:
                st.error("❌ Current: REJECTED")
        else:
            st.warning("⏸️ Current: NOT REVIEWED")
        
        # Overall log stats
        if st.session_state.logger:
            st.divider()
            st.subheader("📈 Overall Stats")
            log_stats = st.session_state.logger.get_stats()
            st.caption(f"Total reviews: {log_stats['total']}")

# Main content
st.markdown("### 🖼️ YOLO Reviewer")

if not st.session_state.pairs:
    st.info("👈 Enter dataset path in sidebar and click Load Dataset to begin")
    st.markdown("""
    ### Quick Start:
    1. Enter your dataset path in the sidebar
    2. Select image size (default: Compact - smaller & faster)
    3. Click **Load Dataset**
    4. Review images and press **A** to Accept or **R** to Reject
    5. App auto-advances to next image
    6. Use **Previous/Next** to navigate manually
    7. All decisions are logged to `review_log.csv`
    
    ### ⌨️ Keyboard Shortcuts:
    - **A** = Accept current image
    - **R** = Reject current image
    - Both auto-advance to next image!
    """)
else:
    idx = st.session_state.current_index
    pair = st.session_state.pairs[idx]
    
    # Compact header - single line
    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
    with col1:
        st.markdown(f"**Image {idx + 1}/{len(st.session_state.pairs)}** • `{os.path.basename(pair['image'])[:40]}...`")
    with col2:
        progress_pct = ((idx + 1) / len(st.session_state.pairs)) * 100
        st.caption(f"📊 {progress_pct:.0f}%")
    with col3:
        if idx in st.session_state.decisions:
            status = st.session_state.decisions[idx]
            if status == 'accepted':
                st.caption("✅ ACCEPTED")
            else:
                st.caption("❌ REJECTED")
        else:
            st.caption("⏸️ NEW")
    with col4:
        # Show processed count
        st.caption(f"📋 Done: {st.session_state.processed_count}")
    
    # === MAIN TWO-COLUMN LAYOUT: IMAGE LEFT, BUTTONS RIGHT ===
    col_img, col_controls = st.columns([3, 1])
    
    # Left: image + annotations
    with col_img:
        max_size = IMAGE_SIZE_PRESETS[st.session_state.image_size_preset]
        img = load_image(pair['image'], max_size)
        if img:
            boxes = parse_yolo_annotation(pair['label'], img.size[0], img.size[1])
            img_display = draw_boxes(img.copy(), boxes, st.session_state.class_names)
            
            # Display image (respects the selected size)
            st.image(img_display)  # uses column width
            
            # Compact annotation info - inline
            if boxes:
                box_info = " • ".join([
                    st.session_state.class_names.get(box['class_id'], f"C{box['class_id']}") 
                    for box in boxes[:5]
                ])
                if len(boxes) > 5:
                    box_info += f" ... +{len(boxes)-5} more"
                st.caption(f"🏷️ {len(boxes)} objects: {box_info}")
            else:
                st.caption("⚠️ No annotations")
    
    # Right: vertical buttons & keyboard hint
    with col_controls:
        st.markdown("#### Actions")
        st.write("")  # small spacer
        
        # Keyboard shortcuts component (still global)
        keyboard_js = """
        <script>
        document.addEventListener('keydown', function(event) {
            // Only trigger if not typing in an input field
            if (event.target.tagName !== 'INPUT' && event.target.tagName !== 'TEXTAREA') {
                if (event.key === 'a' || event.key === 'A') {
                    // Find and click Accept button
                    const buttons = window.parent.document.querySelectorAll('button');
                    buttons.forEach(btn => {
                        if (btn.innerText.includes('ACCEPT')) {
                            btn.click();
                        }
                    });
                } else if (event.key === 'r' || event.key === 'R') {
                    // Find and click Reject button
                    const buttons = window.parent.document.querySelectorAll('button');
                    buttons.forEach(btn => {
                        if (btn.innerText.includes('REJECT')) {
                            btn.click();
                        }
                    });
                }
            }
        });
        </script>
        """
        components.html(keyboard_js, height=0)
        
        # Vertical stack of buttons
        if st.button("✅ ACCEPT (A)", type="primary", use_container_width=True, key="accept"):
            handle_decision('accepted')
            st.rerun()
        
        if st.button("❌ REJECT (R)", use_container_width=True, key="reject"):
            handle_decision('rejected')
            st.rerun()
        
        st.write("")  # spacer
        
        if st.button("⬅️ PREV", use_container_width=True, disabled=(idx == 0)):
            st.session_state.current_index -= 1
            st.rerun()
        
        if st.button("NEXT ➡️", use_container_width=True, disabled=(idx >= len(st.session_state.pairs) - 1)):
            st.session_state.current_index += 1
            st.rerun()
        
        st.caption("⌨️ Press **A** or **R** • Auto-advances")
