# color_logger.py
import time
from typing import Optional, Dict, Any

# Colorful logging
try:
    from colorama import init, Fore, Back, Style
    init(autoreset=True)
    COLORS_AVAILABLE = True
except ImportError:
    # Fallback if colorama not available
    class DummyColor:
        def __getattr__(self, name): return ""
    Fore = Back = Style = DummyColor()
    COLORS_AVAILABLE = False

# Progress bar support
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Fallback tqdm-like class
    class tqdm:
        def __init__(self, *args, **kwargs):
            self.total = kwargs.get('total', 100)
            self.desc = kwargs.get('desc', '')
            self.current = 0
        
        def update(self, n=1):
            self.current += n
            if self.total > 0:
                pct = (self.current / self.total) * 100
                print(f"\r{self.desc}: {pct:.1f}% ({self.current}/{self.total})", end='')
        
        def set_postfix(self, **kwargs):
            pass
        
        def close(self):
            print()  # newline
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            self.close()


class TrainingProgressBar:
    """Enhanced progress bar for training with tqdm integration."""
    
    def __init__(self, total_steps: int, desc: str = "Training", logger=None):
        self.total_steps = total_steps
        self.desc = desc
        self.logger = logger
        self.start_time = time.time()
        
        if TQDM_AVAILABLE:
            self.pbar = tqdm(
                total=total_steps,
                desc=f"{Fore.BLUE}{desc}{Style.RESET_ALL}",
                ncols=100,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}',
                colour='blue'
            )
        else:
            self.pbar = tqdm(total=total_steps, desc=desc)
            if logger:
                logger.warning("⚠️  tqdm not available, using fallback progress display")
    
    def update(self, step: int, **metrics):
        """Update progress bar with current step and metrics."""
        if TQDM_AVAILABLE and hasattr(self.pbar, 'n'):
            # Format metrics (colors may not work well in all terminals with tqdm)
            postfix_dict = {}
            for key, value in metrics.items():
                if isinstance(value, float):
                    # Use simpler formatting for tqdm compatibility
                    postfix_dict[key] = f"{value:.3f}"
                else:
                    postfix_dict[key] = str(value)
            
            self.pbar.set_postfix(**postfix_dict)
            # Update to the current step
            if step > getattr(self.pbar, 'n', 0):
                self.pbar.update(step - getattr(self.pbar, 'n', 0))
        else:
            # Fallback update
            if hasattr(self.pbar, 'current'):
                self.pbar.update(step - self.pbar.current)
            else:
                self.pbar.update(1)
    
    def close(self):
        """Close the progress bar."""
        if hasattr(self.pbar, 'close'):
            self.pbar.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


class ColorLogger:
    """
    A colorful logger for training progress tracking.
    
    Features:
    - Color-coded status messages (INFO, SUCCESS, WARN, ERROR, etc.)
    - Timestamp tracking from initialization
    - Progress bars with percentage completion
    - Smart color coding based on value ranges
    - Graceful fallback when colorama is not available
    - Integration with tqdm for enhanced progress tracking
    """
    
    def __init__(self, name="Logger"):
        self.name = name
        self.start_time = time.time()
        self.progress_bar: Optional[TrainingProgressBar] = None
        
        if COLORS_AVAILABLE:
            self.info(f"🎨 ColorLogger initialized with colorama support")
        else:
            print(f"[{self.name}] ColorLogger initialized (no colors - install colorama for colors)")
        
        if TQDM_AVAILABLE:
            self.info(f"📊 tqdm available for enhanced progress bars")
        else:
            self.warning("⚠️  tqdm not available, using basic progress display")
    
    def _timestamp(self):
        """Get elapsed time since logger initialization."""
        elapsed = time.time() - self.start_time
        return f"{elapsed:.1f}s"
    
    def info(self, msg):
        """General information message in cyan."""
        print(f"{Fore.CYAN}[INFO {self._timestamp()}]{Style.RESET_ALL} {msg}")
    
    def success(self, msg):
        """Success message in green."""
        print(f"{Fore.GREEN}[SUCCESS {self._timestamp()}]{Style.RESET_ALL} {msg}")
    
    def warning(self, msg):
        """Warning message in yellow."""
        print(f"{Fore.YELLOW}[WARN {self._timestamp()}]{Style.RESET_ALL} {msg}")
    
    def error(self, msg):
        """Error message in red."""
        print(f"{Fore.RED}[ERROR {self._timestamp()}]{Style.RESET_ALL} {msg}")
    
    def step(self, step, msg):
        """Training step message in blue."""
        print(f"{Fore.BLUE}[STEP {step:>5} | {self._timestamp()}]{Style.RESET_ALL} {msg}")
    
    def eval(self, step, msg):
        """Evaluation message in magenta."""
        print(f"{Fore.MAGENTA}[EVAL {step:>5} | {self._timestamp()}]{Style.RESET_ALL} {msg}")
    
    def debug(self, msg):
        """Debug message in dim white."""
        print(f"{Style.DIM}[DEBUG {self._timestamp()}]{Style.RESET_ALL} {msg}")
    
    def progress(self, current, total, prefix="Progress", bar_length=30):
        """
        Display a progress bar.
        
        Args:
            current: Current progress value
            total: Total/maximum value
            prefix: Label for the progress bar
            bar_length: Length of the progress bar in characters
        """
        if total > 0:
            pct = (current / total) * 100
            filled = int(bar_length * current / total)
            bar = "█" * filled + "░" * (bar_length - filled)
            print(f"{Fore.CYAN}[{prefix}]{Style.RESET_ALL} {bar} {pct:5.1f}% ({current:,}/{total:,})")
        else:
            print(f"{Fore.CYAN}[{prefix}]{Style.RESET_ALL} No progress (total=0)")
    
    def metric(self, name, value, good_threshold=None, warn_threshold=None, higher_is_better=True):
        """
        Display a metric with automatic color coding.
        
        Args:
            name: Metric name
            value: Metric value
            good_threshold: Threshold for green color
            warn_threshold: Threshold for yellow color
            higher_is_better: If True, higher values get better colors
        """
        color = Fore.WHITE  # default
        
        if good_threshold is not None and warn_threshold is not None:
            if higher_is_better:
                if value >= good_threshold:
                    color = Fore.GREEN
                elif value >= warn_threshold:
                    color = Fore.YELLOW
                else:
                    color = Fore.RED
            else:  # lower is better
                if value <= good_threshold:
                    color = Fore.GREEN
                elif value <= warn_threshold:
                    color = Fore.YELLOW
                else:
                    color = Fore.RED
        
        return f"{name}: {color}{value}{Style.RESET_ALL}"
    
    def loss_display(self, losses_dict, thresholds=None):
        """
        Display multiple losses with color coding.
        
        Args:
            losses_dict: Dict of loss_name -> loss_value
            thresholds: Dict of loss_name -> (good_threshold, warn_threshold)
        """
        if thresholds is None:
            thresholds = {}
        
        loss_strs = []
        for name, value in losses_dict.items():
            good_thresh, warn_thresh = thresholds.get(name, (None, None))
            loss_str = self.metric(name, f"{value:.3f}", good_thresh, warn_thresh, higher_is_better=False)
            loss_strs.append(loss_str)
        
        return " | ".join(loss_strs)
    
    def separator(self, char="=", length=60, title=None):
        """Print a colored separator line with optional title."""
        line = char * length
        if title:
            padding = (length - len(title) - 2) // 2
            line = char * padding + f" {title} " + char * padding
            if len(line) < length:
                line += char
        
        print(f"{Fore.CYAN}{line}{Style.RESET_ALL}")
    
    def summary_table(self, data_dict, title="Summary"):
        """
        Display a summary table.
        
        Args:
            data_dict: Dict of label -> (value, color_or_none)
        """
        self.separator(title=title)
        
        max_label_len = max(len(str(k)) for k in data_dict.keys()) if data_dict else 0
        
        for label, value_info in data_dict.items():
            if isinstance(value_info, tuple):
                value, color = value_info
                if color:
                    print(f"  {label:<{max_label_len}}: {color}{value}{Style.RESET_ALL}")
                else:
                    print(f"  {label:<{max_label_len}}: {value}")
            else:
                print(f"  {label:<{max_label_len}}: {value_info}")
        
        self.separator()
    
    def create_progress_bar(self, total_steps: int, desc: str = "Progress") -> TrainingProgressBar:
        """Create a new training progress bar."""
        if self.progress_bar is not None:
            self.progress_bar.close()
        
        self.progress_bar = TrainingProgressBar(total_steps, desc, logger=self)
        return self.progress_bar
    
    def update_progress(self, step: int, **metrics):
        """Update the current progress bar if it exists."""
        if self.progress_bar is not None:
            self.progress_bar.update(step, **metrics)
    
    def close_progress_bar(self):
        """Close the current progress bar."""
        if self.progress_bar is not None:
            self.progress_bar.close()
            self.progress_bar = None


# Convenience function for quick logger creation
def get_logger(name="Training"):
    """Get a ColorLogger instance with the given name."""
    return ColorLogger(name)


# Pre-defined color schemes for common ML metrics
class MLColors:
    """Common color schemes for machine learning metrics."""
    
    @staticmethod
    def loss_colors(loss_value, good=1.0, warn=2.0):
        """Color code for loss values (lower is better)."""
        if loss_value <= good:
            return Fore.GREEN
        elif loss_value <= warn:
            return Fore.YELLOW
        else:
            return Fore.RED
    
    @staticmethod
    def accuracy_colors(acc_value, good=0.8, warn=0.6):
        """Color code for accuracy values (higher is better)."""
        if acc_value >= good:
            return Fore.GREEN
        elif acc_value >= warn:
            return Fore.YELLOW
        else:
            return Fore.RED
    
    @staticmethod
    def improvement_colors(improvement):
        """Color code for improvement values (positive is good)."""
        if improvement > 0:
            return Fore.GREEN
        elif improvement == 0:
            return Fore.YELLOW
        else:
            return Fore.RED


if __name__ == "__main__":
    # Demo/test the logger
    logger = get_logger("Demo")
    
    logger.info("This is an info message")
    logger.success("This is a success message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.debug("This is a debug message")
    
    logger.step(100, "Training step example")
    logger.eval(100, "Evaluation example")
    
    logger.progress(30, 100, "Training")
    
    # Demo metric display
    print("\nMetric examples:")
    print(logger.metric("Accuracy", 0.85, good_threshold=0.8, warn_threshold=0.6))
    print(logger.metric("Loss", 1.2, good_threshold=1.0, warn_threshold=2.0, higher_is_better=False))
    
    # Demo loss display
    losses = {"SFT": 0.8, "KL": 0.15, "DPO": 1.1}
    thresholds = {"SFT": (1.0, 2.0), "KL": (0.1, 0.5), "DPO": (1.0, 2.0)}
    print(f"\nLosses: {logger.loss_display(losses, thresholds)}")
    
    # Demo summary
    summary_data = {
        "Final Accuracy": ("0.856", Fore.GREEN),
        "Training Time": ("120.5s", None),
        "Best Loss": ("0.823", Fore.YELLOW),
        "Steps": ("5000", None)
    }
    logger.summary_table(summary_data, "Training Results")
