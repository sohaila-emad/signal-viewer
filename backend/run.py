from app import create_app, socketio
import os

app = create_app()

if __name__ == '__main__':
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 5000))
    
    # Run with socketio for websocket support
    socketio.run(
        app,
        host='0.0.0.0',  # Allow external connections
        port=port,
        debug=True,       # Set to False in production
        allow_unsafe_werkzeug=True  # For development
    )
