#!/bin/bash
# Telescope Detection System Service Management Script

SERVICE_NAME="telescope_detection"
SERVICE_FILE="telescope_detection.service"
SYSTEMD_DIR="/etc/systemd/system"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Telescope Detection Service Manager${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

check_root() {
    if [ "$EUID" -ne 0 ]; then
        print_error "This command requires sudo/root privileges"
        exit 1
    fi
}

install_service() {
    print_header
    echo "Installing service..."

    check_root

    # Copy service file to systemd directory
    if [ -f "$PROJECT_DIR/$SERVICE_FILE" ]; then
        cp "$PROJECT_DIR/$SERVICE_FILE" "$SYSTEMD_DIR/$SERVICE_NAME.service"
        print_success "Service file copied to $SYSTEMD_DIR"
    else
        print_error "Service file not found: $PROJECT_DIR/$SERVICE_FILE"
        exit 1
    fi

    # Reload systemd
    systemctl daemon-reload
    print_success "Systemd daemon reloaded"

    # Enable service to start on boot
    systemctl enable "$SERVICE_NAME.service"
    print_success "Service enabled (will start on boot)"

    echo ""
    print_success "Service installed successfully!"
    echo ""
    print_info "Next steps:"
    echo "  • Start service:  sudo $0 start"
    echo "  • Check status:   sudo $0 status"
    echo "  • View logs:      sudo $0 logs"
}

uninstall_service() {
    print_header
    echo "Uninstalling service..."

    check_root

    # Stop service if running
    systemctl stop "$SERVICE_NAME.service" 2>/dev/null

    # Disable service
    systemctl disable "$SERVICE_NAME.service" 2>/dev/null

    # Remove service file
    if [ -f "$SYSTEMD_DIR/$SERVICE_NAME.service" ]; then
        rm "$SYSTEMD_DIR/$SERVICE_NAME.service"
        print_success "Service file removed"
    fi

    # Reload systemd
    systemctl daemon-reload
    print_success "Systemd daemon reloaded"

    print_success "Service uninstalled successfully!"
}

start_service() {
    print_header
    echo "Starting service..."

    check_root

    systemctl start "$SERVICE_NAME.service"

    if [ $? -eq 0 ]; then
        print_success "Service started successfully"
        echo ""
        sleep 2
        show_status
    else
        print_error "Failed to start service"
        exit 1
    fi
}

stop_service() {
    print_header
    echo "Stopping service..."

    check_root

    systemctl stop "$SERVICE_NAME.service"

    if [ $? -eq 0 ]; then
        print_success "Service stopped successfully"
    else
        print_error "Failed to stop service"
        exit 1
    fi
}

restart_service() {
    print_header
    echo "Restarting service..."

    check_root

    systemctl restart "$SERVICE_NAME.service"

    if [ $? -eq 0 ]; then
        print_success "Service restarted successfully"
        echo ""
        sleep 2
        show_status
    else
        print_error "Failed to restart service"
        exit 1
    fi
}

show_status() {
    print_header
    systemctl status "$SERVICE_NAME.service" --no-pager
}

show_logs() {
    print_header
    echo "Showing logs (press Ctrl+C to exit)..."
    echo ""

    if [ "$1" == "follow" ] || [ "$1" == "-f" ]; then
        journalctl -u "$SERVICE_NAME.service" -f
    else
        journalctl -u "$SERVICE_NAME.service" -n 100 --no-pager
    fi
}

enable_service() {
    print_header
    echo "Enabling service to start on boot..."

    check_root

    systemctl enable "$SERVICE_NAME.service"
    print_success "Service enabled"
}

disable_service() {
    print_header
    echo "Disabling service from starting on boot..."

    check_root

    systemctl disable "$SERVICE_NAME.service"
    print_success "Service disabled"
}

show_help() {
    print_header
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  install    Install service (requires sudo)"
    echo "  uninstall  Remove service (requires sudo)"
    echo "  start      Start the service (requires sudo)"
    echo "  stop       Stop the service (requires sudo)"
    echo "  restart    Restart the service (requires sudo)"
    echo "  status     Show service status"
    echo "  logs       Show recent logs"
    echo "  logs -f    Follow logs in real-time"
    echo "  enable     Enable service to start on boot (requires sudo)"
    echo "  disable    Disable service from starting on boot (requires sudo)"
    echo "  help       Show this help message"
    echo ""
    echo "Examples:"
    echo "  sudo $0 install    # First time setup"
    echo "  sudo $0 start      # Start the service"
    echo "  $0 status          # Check if running"
    echo "  $0 logs -f         # Watch logs live"
    echo ""
}

# Main script logic
case "$1" in
    install)
        install_service
        ;;
    uninstall)
        uninstall_service
        ;;
    start)
        start_service
        ;;
    stop)
        stop_service
        ;;
    restart)
        restart_service
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs "$2"
        ;;
    enable)
        enable_service
        ;;
    disable)
        disable_service
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        show_help
        exit 1
        ;;
esac
