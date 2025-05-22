import subprocess
import sys
import signal

def caffeinate_system():
  try:
    # Start caffeinate process to prevent sleep
    process = subprocess.Popen(['caffeinate', '-i', '-d'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("System will stay awake. Press Ctrl+C to exit.")
    
    # Keep the script running until interrupted
    process.wait()
  
  except KeyboardInterrupt:
    # Handle Ctrl+C gracefully
    process.terminate()
    print("\nStopping caffeinate, system can sleep normally now.")
    sys.exit(0)

if __name__ == "__main__":
  # Set up signal handler for clean exit
  signal.signal(signal.SIGINT, lambda x, y: None)
  caffeinate_system()