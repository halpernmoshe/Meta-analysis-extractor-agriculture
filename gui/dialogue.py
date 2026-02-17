"""
GUI Dialogue module for Meta-Analysis Extraction System

CLI-based version - uses terminal input/output instead of tkinter popups.
"""
import sys
from typing import List, Optional, Dict, Any


class DialogueManager:
    """Manages dialogues for user interaction via CLI"""

    def __init__(self):
        pass

    def show_info(self, title: str, message: str):
        print(f"\n{'='*60}")
        print(f"[INFO] {title}")
        print(f"{'='*60}")
        print(message)
        print(f"{'='*60}\n")

    def show_warning(self, title: str, message: str):
        print(f"\n{'='*60}")
        print(f"[WARNING] {title}")
        print(f"{'='*60}")
        print(message)
        print(f"{'='*60}\n")

    def show_error(self, title: str, message: str):
        print(f"\n{'='*60}")
        print(f"[ERROR] {title}")
        print(f"{'='*60}")
        print(message)
        print(f"{'='*60}\n")

    def ask_confirm(self, title: str, message: str) -> bool:
        print(f"\n{'='*60}")
        print(f"[CONFIRM] {title}")
        print(f"{'='*60}")
        print(message)
        print(f"{'='*60}")
        while True:
            response = input("Continue? (y/n): ").strip().lower()
            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no']:
                return False
            print("Please enter 'y' or 'n'")

    def ask_directory(self, title: str = "Select Directory") -> Optional[str]:
        print(f"\n[{title}]")
        path = input("Enter directory path (or press Enter to cancel): ").strip()
        return path if path else None

    def ask_choice(self, title: str, question: str, options: List[str],
                   context: str = None, allow_custom: bool = False) -> Optional[str]:
        print(f"\n{'='*60}")
        print(f"[CHOICE] {title}")
        print(f"{'='*60}")
        print(f"\n{question}\n")

        if context:
            print(f"Context: {context}\n")

        print("Options:")
        for i, opt in enumerate(options, 1):
            print(f"  {i}. {opt}")

        if allow_custom:
            print(f"  {len(options)+1}. [Enter custom value]")

        print(f"  0. Cancel")
        print()

        while True:
            try:
                choice = input("Enter number: ").strip()
                if choice == '0':
                    return None

                idx = int(choice) - 1
                if 0 <= idx < len(options):
                    return options[idx]
                elif allow_custom and idx == len(options):
                    custom = input("Enter custom value: ").strip()
                    return custom if custom else None
                else:
                    print(f"Please enter a number between 0 and {len(options) + (1 if allow_custom else 0)}")
            except ValueError:
                print("Please enter a valid number")

    def ask_multiselect(self, title: str, question: str, options: List[str],
                        context: str = None, min_select: int = 0, max_select: int = None,
                        preselected: List[str] = None) -> List[str]:
        print(f"\n{'='*60}")
        print(f"[MULTI-SELECT] {title}")
        print(f"{'='*60}")
        print(f"\n{question}\n")

        if context:
            print(f"Context: {context}\n")

        constraints = []
        if min_select > 0:
            constraints.append(f"at least {min_select}")
        if max_select:
            constraints.append(f"at most {max_select}")
        if constraints:
            print(f"(Select {' and '.join(constraints)} option(s))\n")

        print("Options:")
        for i, opt in enumerate(options, 1):
            marker = "*" if preselected and opt in preselected else " "
            print(f"  {i}. [{marker}] {opt}")

        print()
        print("Enter numbers separated by commas (e.g., 1,3,5)")
        print("Or 'all' to select all, '0' to cancel")
        print()

        while True:
            response = input("Your selection: ").strip().lower()

            if response == '0':
                return []

            if response == 'all':
                if max_select and len(options) > max_select:
                    print(f"Cannot select all - maximum is {max_select}")
                    continue
                return options.copy()

            try:
                indices = [int(x.strip()) - 1 for x in response.split(',') if x.strip()]
                selected = []
                valid = True

                for idx in indices:
                    if 0 <= idx < len(options):
                        if options[idx] not in selected:
                            selected.append(options[idx])
                    else:
                        print(f"Invalid option number: {idx + 1}")
                        valid = False
                        break

                if not valid:
                    continue

                if len(selected) < min_select:
                    print(f"Please select at least {min_select} option(s)")
                    continue

                if max_select and len(selected) > max_select:
                    print(f"Please select at most {max_select} option(s)")
                    continue

                return selected

            except ValueError:
                print("Please enter valid numbers separated by commas")

    def ask_text(self, title: str, question: str, context: str = None,
                 default: str = "", multiline: bool = False) -> Optional[str]:
        print(f"\n{'='*60}")
        print(f"[INPUT] {title}")
        print(f"{'='*60}")
        print(f"\n{question}\n")

        if context:
            print(f"Context: {context}\n")

        if default:
            print(f"Default: {default}")

        if multiline:
            print("(Enter text, then type 'END' on a new line to finish, or 'CANCEL' to cancel)")
            lines = []
            while True:
                line = input()
                if line.strip().upper() == 'END':
                    break
                if line.strip().upper() == 'CANCEL':
                    return None
                lines.append(line)
            return '\n'.join(lines) if lines else default
        else:
            response = input(f"Enter value (or press Enter for default): ").strip()
            return response if response else default

    def show_summary(self, title: str, summary: Dict[str, Any]):
        print(f"\n{'='*60}")
        print(f"[SUMMARY] {title}")
        print(f"{'='*60}")
        self._print_dict(summary, indent=0)
        print(f"{'='*60}")
        input("Press Enter to continue...")

    def _print_dict(self, d: Dict[str, Any], indent: int = 0):
        prefix = "  " * indent
        for key, value in d.items():
            if isinstance(value, dict):
                print(f"{prefix}{key}:")
                self._print_dict(value, indent + 1)
            elif isinstance(value, list):
                print(f"{prefix}{key}:")
                for item in value:
                    if isinstance(item, dict):
                        self._print_dict(item, indent + 1)
                    else:
                        print(f"{prefix}  - {item}")
            else:
                print(f"{prefix}{key}: {value}")

    def show_progress(self, title: str, message: str, progress: float = 0,
                      status: str = "") -> 'ProgressWindow':
        return ProgressWindow(title, message, progress, status)


class ProgressWindow:
    """CLI-based progress indicator"""

    def __init__(self, title: str, message: str, progress: float = 0, status: str = ""):
        self.title = title
        self.message = message
        self.progress = progress
        self.status = status
        self._last_line_length = 0
        print(f"\n[{title}] {message}")
        self._print_progress()

    def _print_progress(self):
        bar_length = 40
        filled = int(bar_length * self.progress / 100)
        bar = '=' * filled + '-' * (bar_length - filled)
        line = f"\r  [{bar}] {self.progress:.0f}% - {self.status}"
        # Pad with spaces to clear previous longer lines
        line = line.ljust(self._last_line_length)
        self._last_line_length = len(line)
        print(line, end='', flush=True)

    def update(self, progress: float = None, status: str = None, message: str = None):
        if progress is not None:
            self.progress = progress
        if status is not None:
            self.status = status
        if message is not None:
            self.message = message
        self._print_progress()

    def close(self):
        print()  # New line after progress bar
        print(f"[{self.title}] Complete\n")
