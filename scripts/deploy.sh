#!/bin/bash

# Quick Production Deployment Script for MoneyVerse AI Backend
# This script automates the deployment process

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
APP_NAME="moneyverse-ai-backend"
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

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

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   log_error "This script should not be run as root"
   exit 1
fi

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is required but not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is required but not installed"
        exit 1
    fi
    
    # Check if .env file exists
    if [[ ! -f "$APP_DIR/.env" ]]; then
        log_warning "No .env file found. Creating from template..."
        if [[ -f "$APP_DIR/production.env" ]]; then
            cp "$APP_DIR/production.env" "$APP_DIR/.env"
            log_warning "Please edit .env file with your production values before continuing"
            exit 1
        else
            log_error "No environment template found"
            exit 1
        fi
    fi
    
    log_success "Prerequisites check completed"
}

# Deploy with Docker Compose
deploy_docker() {
    log_info "Deploying with Docker Compose..."
    
    cd "$APP_DIR"
    
    # Stop existing containers
    log_info "Stopping existing containers..."
    docker-compose -f docker-compose.production.yml down || true
    
    # Pull latest changes
    log_info "Pulling latest changes..."
    git pull origin main || log_warning "Could not pull latest changes"
    
    # Build and start services
    log_info "Building and starting services..."
    docker-compose -f docker-compose.production.yml up -d --build
    
    # Wait for services to be ready
    log_info "Waiting for services to be ready..."
    sleep 30
    
    # Check service health
    log_info "Checking service health..."
    docker-compose -f docker-compose.production.yml ps
    
    log_success "Docker deployment completed"
}

# Deploy manually
deploy_manual() {
    log_info "Deploying manually..."
    
    cd "$APP_DIR"
    
    # Pull latest changes
    log_info "Pulling latest changes..."
    git pull origin main || log_warning "Could not pull latest changes"
    
    # Install dependencies
    log_info "Installing dependencies..."
    pip install -r requirements.txt
    
    # Run database migrations
    log_info "Running database migrations..."
    alembic upgrade head
    
    # Start application
    log_info "Starting application..."
    ./scripts/start_production.sh restart
    
    log_success "Manual deployment completed"
}

# Check deployment
check_deployment() {
    log_info "Checking deployment..."
    
    # Wait for application to be ready
    local max_attempts=30
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        if curl -f -s "http://localhost:8000/health" > /dev/null 2>&1; then
            log_success "Application is healthy and responding"
            
            # Show health details
            echo "Health Details:"
            curl -s "http://localhost:8000/health/detailed" | jq . 2>/dev/null || curl -s "http://localhost:8000/health/detailed"
            
            return 0
        fi
        
        log_info "Health check attempt $attempt/$max_attempts failed, retrying in 2 seconds..."
        sleep 2
        ((attempt++))
    done
    
    log_error "Deployment health check failed"
    return 1
}

# Show deployment status
show_status() {
    log_info "Deployment Status:"
    
    echo "Service Status:"
    docker-compose -f docker-compose.production.yml ps 2>/dev/null || echo "Docker Compose not available"
    
    echo ""
    echo "Application Health:"
    curl -s "http://localhost:8000/health" 2>/dev/null || echo "Application not responding"
    
    echo ""
    echo "Recent Logs:"
    docker-compose -f docker-compose.production.yml logs --tail=10 backend 2>/dev/null || echo "No logs available"
}

# Main execution
main() {
    log_info "Starting MoneyVerse AI Backend Deployment"
    log_info "Application directory: $APP_DIR"
    
    case "${1:-docker}" in
        "docker")
            check_prerequisites
            deploy_docker
            check_deployment
            show_status
            ;;
        "manual")
            check_prerequisites
            deploy_manual
            check_deployment
            show_status
            ;;
        "status")
            show_status
            ;;
        "logs")
            log_info "Showing recent logs..."
            docker-compose -f docker-compose.production.yml logs -f backend
            ;;
        "stop")
            log_info "Stopping deployment..."
            docker-compose -f docker-compose.production.yml down
            log_success "Deployment stopped"
            ;;
        "restart")
            log_info "Restarting deployment..."
            docker-compose -f docker-compose.production.yml restart
            log_success "Deployment restarted"
            ;;
        *)
            echo "Usage: $0 {docker|manual|status|logs|stop|restart}"
            echo ""
            echo "Commands:"
            echo "  docker   - Deploy using Docker Compose (default)"
            echo "  manual   - Deploy manually without Docker"
            echo "  status   - Show deployment status"
            echo "  logs     - Show application logs"
            echo "  stop     - Stop deployment"
            echo "  restart  - Restart deployment"
            exit 1
            ;;
    esac
}

# Run main function
main "$@" 