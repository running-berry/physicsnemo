#!/bin/bash

# Configuration variables
SOURCE_USER=""
SOURCE_HOST=""
SOURCE_BASE_PATH=""

JUMP_HOST="jb1.frontier.nvidia.com"
JUMP_PORT="2222"
JUMP_USER=""

TARGET_USER=""
TARGET_HOST=""
TARGET_BASE_PATH=""

SSH_KEY="~/.ssh/id_ed25519"
LOCAL_TEMP_DIR="."
ITEM_TYPE="folder"  
CLEANUP=true

# Function to display usage
usage() {
    echo "Usage: $0 <item_path> [options]"
    echo ""
    echo "Arguments:"
    echo "  item_path                  Path to item to transfer (relative to source base path)"
    echo ""
    echo "Options:"
    echo "  --type TYPE                Item type: 'file' or 'folder' (default: $ITEM_TYPE)"
    echo "  -s, --source-path PATH     Source base path (default: $SOURCE_BASE_PATH)"
    echo "  -t, --target-path PATH     Target base path (default: $TARGET_BASE_PATH)"
    echo "  -l, --local-dir DIR        Local temporary directory (default: $LOCAL_TEMP_DIR)"
    echo "  --source-user USER         Source user (default: $SOURCE_USER)"
    echo "  --source-host HOST         Source host (default: $SOURCE_HOST)"
    echo "  --target-user USER         Target user (default: $TARGET_USER)"
    echo "  --target-host HOST         Target host (default: $TARGET_HOST)"
    echo "  --jump-user USER           Jump host user (default: $JUMP_USER)"
    echo "  --jump-host HOST           Jump host (default: $JUMP_HOST)"
    echo "  --ssh-key PATH             SSH key path (default: $SSH_KEY)"
    echo "  --no-cleanup               Keep local files after transfer"
    echo "  -h, --help                 Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Transfer folders (default behavior)"
    echo "  $0 ERA5_nc/t850"
    echo "  $0 ERA5_nc --type folder"
    echo ""
    echo "  # Transfer files"
    echo "  $0 data/file.nc --type file"
    echo "  $0 scripts/process.sh --type file"
    echo ""
    echo "  # With custom options"
    echo "  $0 ERA5_nc/u10 --target-path /custom/path --type folder"
    echo "  $0 myfile.txt --type file --local-dir /tmp --no-cleanup"
    exit 1
}

# Function to transfer data
transfer_item() {
    local item_path="$1"
    
    if [ -z "$item_path" ]; then
        echo "Error: Item path is required"
        usage
    fi
    
    # Validate item type
    if [ "$ITEM_TYPE" != "file" ] && [ "$ITEM_TYPE" != "folder" ]; then
        echo "Error: --type must be 'file' or 'folder'"
        return 1
    fi
    
    local item_name=$(basename "$item_path")
    local full_source_path="${SOURCE_BASE_PATH}/${item_path}"
    
    echo "=== Transfer Configuration ==="
    echo "Item: $item_path"
    echo "Type: $ITEM_TYPE"
    echo "Source: ${SOURCE_USER}@${SOURCE_HOST}:${full_source_path}"
    echo "Target: ${TARGET_USER}@${TARGET_HOST}:${TARGET_BASE_PATH}"
    echo "Local temp: ${LOCAL_TEMP_DIR}"
    echo "Cleanup: $CLEANUP"
    echo "=============================="
    
    # Step 1: Download from source
    echo "Step 1: Downloading $ITEM_TYPE from source..."
    local scp_options=""
    if [ "$ITEM_TYPE" = "folder" ]; then
        scp_options="-r"
    fi
    
    if ! scp $scp_options "${SOURCE_USER}@${SOURCE_HOST}:${full_source_path}" "${LOCAL_TEMP_DIR}/"; then
        echo "Error: Failed to download $ITEM_TYPE from source"
        return 1
    fi
    
    # Step 2: Upload to target through jump host
    echo "Step 2: Uploading $ITEM_TYPE to target..."
    if ! rsync -azv --progress --partial \
        -e "ssh -i ${SSH_KEY} -o \"ProxyCommand=ssh -i ${SSH_KEY} -W %h:%p ${JUMP_USER}@${JUMP_HOST} -p ${JUMP_PORT}\"" \
        "${LOCAL_TEMP_DIR}/${item_name}" \
        "${TARGET_USER}@${TARGET_HOST}:${TARGET_BASE_PATH}/"; then
        echo "Error: Failed to upload $ITEM_TYPE to target"
        return 1
    fi
    
    # Step 3: Clean up local files (optional)
    if [ "$CLEANUP" = true ]; then
        echo "Step 3: Cleaning up local files..."
        if ! rm -rf "${LOCAL_TEMP_DIR}/${item_name}"; then
            echo "Warning: Failed to remove local files"
            return 1
        fi
    else
        echo "Step 3: Skipping cleanup (local files preserved at: ${LOCAL_TEMP_DIR}/${item_name})"
    fi
    
    echo "Transfer of $ITEM_TYPE '${item_path}' completed successfully!"
}

# Parse command line arguments
ITEM_PATH=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --type)
            ITEM_TYPE="$2"
            shift 2
            ;;
        -s|--source-path)
            SOURCE_BASE_PATH="$2"
            shift 2
            ;;
        -t|--target-path)
            TARGET_BASE_PATH="$2"
            shift 2
            ;;
        -l|--local-dir)
            LOCAL_TEMP_DIR="$2"
            shift 2
            ;;
        --source-user)
            SOURCE_USER="$2"
            shift 2
            ;;
        --source-host)
            SOURCE_HOST="$2"
            shift 2
            ;;
        --target-user)
            TARGET_USER="$2"
            shift 2
            ;;
        --target-host)
            TARGET_HOST="$2"
            shift 2
            ;;
        --jump-user)
            JUMP_USER="$2"
            shift 2
            ;;
        --jump-host)
            JUMP_HOST="$2"
            shift 2
            ;;
        --ssh-key)
            SSH_KEY="$2"
            shift 2
            ;;
        --no-cleanup)
            CLEANUP=false
            shift
            ;;
        -h|--help)
            usage
            ;;
        -*)
            echo "Unknown option $1"
            usage
            ;;
        *)
            if [ -z "$ITEM_PATH" ]; then
                ITEM_PATH="$1"
            else
                echo "Error: Multiple item paths specified. Only one item can be transferred at a time."
                usage
            fi
            shift
            ;;
    esac
done

# Main execution
if [ -z "$ITEM_PATH" ]; then
    echo "Error: Item path is required"
    usage
fi

transfer_item "$ITEM_PATH"