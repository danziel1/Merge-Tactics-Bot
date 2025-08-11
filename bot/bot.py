import pygetwindow as gw
import pyautogui
import time
import os
import random
import cv2
import numpy as np
import shutil
from skimage.metrics import structural_similarity as ssim
from ultralytics import YOLO




# https://github.com/krazyness/CRBot-public/tree/main?tab=readme-ov-file
def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)     # Grayscale
    img = cv2.GaussianBlur(img, (3, 3), 0)          # Blur noise
    img = cv2.resize(img, (128, 128))               # Normalize size
    return img



class Bot:
    def __init__(self):
        window = gw.getWindowsWithTitle('BlueStacks')[0]
        print(f"Left: {window.left}, Top: {window.top}, Width: {window.width}, Height: {window.height}")

        self.TOP_LEFT_X = window.left
        self.TOP_LEFT_Y = window.top
        self.BOTTOM_RIGHT_X = window.left+window.width
        self.BOTTOM_RIGHT_Y = window.top+window.height
        
        self.WIDTH = self.BOTTOM_RIGHT_X - self.TOP_LEFT_X
        self.HEIGHT = self.BOTTOM_RIGHT_Y - self.TOP_LEFT_Y

        self.model = YOLO("./bot/best.pt")


        self.screenshot_path = "./bot/screenshots"
        self.clear_screenshots()

        os.makedirs(self.screenshot_path, exist_ok=True)
        self.screenshot = None

    
    def clear_screenshots(self):
        try:
            shutil.rmtree(self.screenshot_path)
        except:
            pass
        print("Cleared screenshots")


    def join_game(self):
        pyautogui.moveTo((self.TOP_LEFT_X+self.BOTTOM_RIGHT_X)/2, (self.TOP_LEFT_Y+self.BOTTOM_RIGHT_Y)*0.75, .2)
        time.sleep(0.5)
        pyautogui.click()


    def check_game_start(self):
        # self.clear_screenshots()
        x1 = self.TOP_LEFT_X + self.BOTTOM_RIGHT_X / 15.5
        y1 = self.TOP_LEFT_Y + self.BOTTOM_RIGHT_Y / 1.3
        x2 = self.TOP_LEFT_X + self.BOTTOM_RIGHT_X / 2.75
        y2 = self.TOP_LEFT_Y + self.BOTTOM_RIGHT_Y / 1.1
        left   = round(x1)
        top    = round(y1)
        width  = round(x2 - x1)
        height = round(y2 - y1)
        region = (left, top, width, height)

        screenshot = np.array(pyautogui.screenshot(region=region))
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)


        template = cv2.imread("./templates/cards.png")
        result = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)

        threshold = 0.4
        loc = np.where(result >= threshold)
        if len(loc[0]) > 0:
            return True
        else:
            return False

    def take_screenshot(self):
        x1 = self.TOP_LEFT_X
        y1 = self.TOP_LEFT_Y
        x2 = self.BOTTOM_RIGHT_X
        y2 = self.BOTTOM_RIGHT_Y

        left   = round(x1)
        top    = round(y1)
        width  = round(x2 - x1)
        height = round(y2 - y1)
        region = (left, top, width, height)

        screenshot = pyautogui.screenshot(region=region)
        saveFile = f"{self.screenshot_path}/{random.randint(111111, 9999999999)}.png"
        screenshot.save(saveFile)

        self.screenshot = saveFile
        print(f"Saved screenshot to {saveFile}")
        
        results = self.model.predict(source=self.screenshot, conf=0.5)
        boxes = results[0].boxes
        # results[0].show() 

        self.cards = []
        for i, box in enumerate(boxes):
            xyxy = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = results[0].names[cls_id]

            xMid = round((xyxy[0] + xyxy[2]) / 2, 1)
            yMid = round((xyxy[1] + xyxy[3]) / 2, 1)
            data = {
                "name": label,
                "x": xMid,
                "y": yMid
            }
            self.cards.append(data)


        print(self.cards)
        print(f"{len(self.cards)} cards found")
        return self.cards

    def check_game_end(self):
        pass

    def get_hand(self):
        results = self.model.predict(source=self.screenshot, conf=0.3)
        boxes = results[0].boxes
        # results[0].show() 

        self.cards = []

        for i, box in enumerate(boxes):
            xyxy = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = results[0].names[cls_id]

            xMid = round((xyxy[0] + xyxy[2]) / 2, 1)
            yMid = round((xyxy[1] + xyxy[3]) / 2, 1)
            if yMid > 1230:
                print(f"Detection {i+1}:")
                print(f"  Class: {label} (ID: {cls_id})")
                print(f"  Confidence: {conf:.2f}")
                print(f"  Center: x={xMid}, y={yMid}")
                print()

                data = {
                    "name": label,
                    "x": xMid,
                    "y": yMid
                }
                self.cards.append(data)

        self.cards = sorted(self.cards, key=lambda c: c["x"])

        print(self.cards)
        print(f"Cards found in hand: {len(self.cards)}")
        return self.cards


def offset(item):
    return item["x"] + bot.TOP_LEFT_X, item["y"] + bot.TOP_LEFT_Y

def findTraits(cards):
    clan = 0
    noble = 0
    goblin = 0
    undead = 0
    ace = 0
    juggernaut = 0
    rangers = 0
    assassin = 0
    throwers = 0
    brawler = 0
    for card in cards:
        if card == "knight":
            noble += 1
            juggernaut += 1
        if card == "archer":
            clan += 1
            rangers += 1
        if card == "goblin":
            goblin += 1
            assassin += 1
        if card == "spear":
            goblin += 1
            throwers += 1
        if card == "bomber":
            undead += 1
            throwers += 1
        if card == "barbarian":
            clan += 1
            brawler += 1
        if card == "valkyarie":
            clan += 1
            avenger += 1
        if card == "pekka":
            ace += 1
            juggernaut += 1
        if card == "prince":
            noble += 1
            brawler += 1
        if card == "giant":
            undead += 1
            brawler += 1
        if card == "dart":
            goblin += 1
            rangers += 1
        if card == "executioner":
            ace += 1
            throwers += 1


# arena < 940
# bench 970 - 1025
# upgrade 1177-1197
# shop > 1230

time.sleep(1)
bot = Bot()
# bot.join_game()
# while bot.check_game_start() == False:
#     time.sleep(1)
print("Joined Game")
# while bot.check_game_end() == False:
while 1:
    detections = bot.take_screenshot()

    orbsFound = []
    upgrade = None
    hand = []

    for item in detections:
        if item["name"] == "orb":
            x,y = offset(item)
            orbsFound.append((x,y))

        if item["name"] == "upgrade":
            upgrade = item


    for orb in orbsFound:
        pyautogui.moveTo(orb[0],orb[1])
        time.sleep(0.1)
        pyautogui.click()
        print("Claimed orb")
        time.sleep(0.5)
    if upgrade:
        x, y = offset(upgrade)
        pyautogui.moveTo(x,y)
        time.sleep(0.1)
        pyautogui.click()
        print("Bought upgrade")
            

    input("Press Enter")
    time.sleep(1)

