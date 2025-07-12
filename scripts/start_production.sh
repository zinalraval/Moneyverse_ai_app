#!/bin/bash

# Production Startup Script for MoneyVerse AI Backend
# This script sets up and starts the application in production mode

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
APP_NAME="moneyverse-ai-backend"
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$APP_DIR/logs"
PID_FILE="$APP_DIR/moneyverse.pid"
ENV_FILE="$APP_DIR/production.env"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_dependencies() {
    log_info "Checking dependencies..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is required but not installed"
        exit 1
    fi
    
    # Check pip
    if ! command -v pip3 &> /dev/null; then
        log_error "pip3 is required but not installed"
        exit 1
    fi
    
    # Check virtual environment
    if [[ -z "$VIRTUAL_ENV" ]]; then
        log_warning "No virtual environment detected. Creating one..."
        python3 -m venv venv
        source venv/bin/activate
    fi
    
    log_success "Dependencies check completed"
}

setup_environment() {
    log_info "Setting up environment..."
    
    # Create log directory
    mkdir -p "$LOG_DIR"
    
    # Load environment variables
    if [[ -f "$ENV_FILE" ]]; then
        export $(cat "$ENV_FILE" | grep -v '^#' | xargs)
        log_success "Environment variables loaded from $ENV_FILE"
    else
        log_warning "Production environment file not found: $ENV_FILE"
        log_info "Using default environment variables"
    fi
    
    # Set production environment
    export ENVIRONMENT=production
    export PRODUCTION=true
    export DEBUG=false
    
    log_success "Environment setup completed"
}

install_dependencies() {
    log_info "Installing Python dependencies..."
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install requirements
    pip install -r requirements.txt
    
    log_success "Dependencies installed"
}

setup_database() {
    log_info "Setting up database..."
    
    # Run database migrations
    cd "$APP_DIR"
    alembic upgrade head
    
    log_success "Database setup completed"
}

start_application() {
    log_info "Starting MoneyVerse AI Backend..."
    
    # Check if already running
    if [[ -f "$PID_FILE" ]]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            log_warning "Application is already running with PID $PID"
            return 0
        else
            log_warning "Removing stale PID file"
            rm -f "$PID_FILE"
        fi
    fi
    
    # Start the application with Gunicorn
    cd "$APP_DIR"
    
    # Create log files
    touch "$LOG_DIR/app.log"
    touch "$LOG_DIR/access.log"
    touch "$LOG_DIR/error.log"
    
    # Start with Gunicorn
    gunicorn \
        --bind "$HOST:$PORT" \
        --workers "$WORKERS" \
        --worker-class uvicorn.workers.UvicornWorker \
        --pid "$PID_FILE" \
        --daemon \
        --access-logfile "$LOG_DIR/access.log" \
        --error-logfile "$LOG_DIR/error.log" \
        --log-level info \
        --timeout 120 \
        --keep-alive 5 \
        --max-requests 1000 \
        --max-requests-jitter 100 \
        --preload \
        app.main:app
    
    # Wait for startup
    sleep 3
    
    # Check if started successfully
    if [[ -f "$PID_FILE" ]]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            log_success "Application started successfully with PID $PID"
            log_info "Access logs: $LOG_DIR/access.log"
            log_info "Error logs: $LOG_DIR/error.log"
            log_info "Application logs: $LOG_DIR/app.log"
        else
            log_error "Application failed to start"
            exit 1
        fi
    else
        log_error "PID file not created"
        exit 1
    fi
}

check_health() {
    log_info "Checking application health..."
    
    # Wait for application to be ready
    local max_attempts=30
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        if curl -f -s "http://$HOST:$PORT/health" > /dev/null 2>&1; then
            log_success "Application is healthy and responding"
            return 0
        fi
        
        log_info "Health check attempt $attempt/$max_attempts failed, retrying in 2 seconds..."
        sleep 2
        ((attempt++))
    done
    
    log_error "Application health check failed after $max_attempts attempts"
    return 1
}

show_status() {
    log_info "Application Status:"
    
    if [[ -f "$PID_FILE" ]]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            log_success "✓ Application is running (PID: $PID)"
            
            # Show process info
            echo "Process Information:"
            ps -p "$PID" -o pid,ppid,cmd,etime,pcpu,pmem
            
            # Show port usage
            echo "Port Usage:"
            netstat -tlnp | grep ":$PORT " || echo "Port $PORT not found in netstat"
            
            # Show recent logs
            echo "Recent Application Logs:"
            tail -n 10 "$LOG_DIR/app.log" 2>/dev/null || echo "No application logs found"
            
        else
            log_error "✗ Application is not running (stale PID file)"
        fi
    else
        log_error "✗ Application is not running (no PID file)"
    fi
}

# Main execution
main() {
    log_info "Starting MoneyVerse AI Backend Production Setup"
    log_info "Application directory: $APP_DIR"
    
    case "${1:-start}" in
        "start")
            check_dependencies
            setup_environment
            install_dependencies
            setup_database
            start_application
            check_health
            show_status
            ;;
        "stop")
            if [[ -f "$PID_FILE" ]]; then
                PID=$(cat "$PID_FILE")
                log_info "Stopping application (PID: $PID)..."
                kill "$PID" 2>/dev/null || true
                rm -f "$PID_FILE"
                log_success "Application stopped"
            else
                log_warning "No PID file found"
            fi
            ;;
        "restart")
            log_info "Restarting application..."
            "$0" stop
            sleep 2
            "$0" start
            ;;
        "status")
            show_status
            ;;
        "logs")
            log_info "Showing recent logs..."
            tail -f "$LOG_DIR/app.log" "$LOG_DIR/access.log" "$LOG_DIR/error.log"
            ;;
        "health")
            check_health
            ;;
        *)
            echo "Usage: $0 {start|stop|restart|status|logs|health}"
            echo ""
            echo "Commands:"
            echo "  start   - Start the application"
            echo "  stop    - Stop the application"
            echo "  restart - Restart the application"
            echo "  status  - Show application status"
            echo "  logs    - Show application logs"
            echo "  health  - Check application health"
            exit 1
            ;;
    esac
}

# Run main function
main "$@" 