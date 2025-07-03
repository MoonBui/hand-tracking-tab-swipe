import pyautogui

def register_key(direction: str):
    """
    Register a key press based on the swipe direction.
    """
    match (direction):
        case "right" | "down-right" | "up-right":
            pyautogui.hotkey("ctrl", "tab")
            print("right")
        case "left" | "down-left" | "up-left":
            pyautogui.hotkey("ctrl", "shift", "tab")
            print("left")
        case "up":
            pyautogui.press("up")
            print("up")
        case "down":
            pyautogui.press("down")
            print("down")
        case _:
            print("no direction")