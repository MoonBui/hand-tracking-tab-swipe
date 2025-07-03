import pyautogui

def register_key(direction: str):
    """
    Register a key press based on the swipe direction.
    """
    match (direction):
        case "right" | "down-right" | "up-right":
            pyautogui.press("right")
            print("right")
        case "left" | "down-left" | "up-left":
            pyautogui.press("left")
            print("left")
        case "up":
            pyautogui.press("up")
            print("up")
        case "down":
            pyautogui.press("down")
            print("down")
        case _:
            print("no direction")