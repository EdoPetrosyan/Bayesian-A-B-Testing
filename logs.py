import logging

class CustomFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (line: %(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        
        # Modify log message to include bandit properties
        if hasattr(record, 'bandit'):
            bandit_info = f"Bandit with True Win Rate {record.bandit.p} - Pulled {record.bandit.N} times - Estimated average reward: {round(record.bandit.p_estimate, 4)} - Estimated average regret: {round(record.bandit.r_estimate, 4)}"
            record.msg = f"{record.msg}\n{bandit_info}"
        
        return formatter.format(record)
