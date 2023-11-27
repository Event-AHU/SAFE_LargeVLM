import os
import numpy as np
import random
import pickle
import csv
from easydict import EasyDict
from scipy.io import loadmat
import glob
from sentence_transformers import SentenceTransformer

np.random.seed(0)
random.seed(0)

#attr_words = ['L type clamp back','wink_ Blink an eye','W type clamp back','YTW Rowing Action','Three Step Run','Upper and Lower Swing Arms','9th set of broadcast gymnastics stretching exercises','9th set of broadcast gymnastics side movements','9th set of broadcast gymnastics full body exercise','9th set of broadcast gymnastics chest expansion exercise','9th set of broadcast gymnastics organizing exercises','9th set of broadcast gymnastics jump sports','9th set of broadcast gymnastics kicking exercises','Alternate Front Kick','Alternate Knee Kick to Tip','Alternate Front Kick Jump','Alternate Knee Lift','Alternating Knee Strike','Stretch','Head down to play with phone','Body flexion and rotation','Body Rotation Movement','Side Lift','Side Lift Swivel Arm','Side Lunge Squat','Bow Side Flat Lift','Bow Rowing Action','Bow Straight Arm Rowing','Pouring Water','Make gestures','Make a Face','eavesdropping action','Vigorous Jump','Cutting Vegetables','Scratch Fist','Brushing teeth','Front Fold','Front and rear foot pads','Forward and backward sliding steps','Cutting Nails','Eye Makeup','One handed cheek support','Single Leg Jump','Single Knee Squat','Volume Data Cable','Press stapler','In Place Wide Distance Run','In Place Hip Kick','Jumping Rope in Place','Step in Place','Cross your waist and raise your fist','Cross waist punch','Cross waist circle eyes','Cross legged Spreading Hands','Cross your waist and touch your ears','Holding cheeks with both hands','Stacking Paper','Folding Clothes','Folding Quilt','Percussion','Right swag','Right Shoulder Stretch','Right humeral triple extension','Right Back Stretch','Right Neck Stretch','Right single swing arm','Right thigh front stretch','Right calf stretch','Right Lunge Twist Stretch','Right Bend','Raise hand with right hand','Right hand raised','Right Hand Circle','Cover the top of the head with the right hand','Right Oblique Pull Down Half Squat','Right Waist Stretch','Right Leg Back Pressure','Right iliopsoas muscle stretch','Eating','Same Hand, Same Foot','Tongue sticking out','Shout','Biting Lips','Drinking Water','Spray Flower Grass','hiss action','Sitting on a stool and rotating in circles','Ear blocking','Thigh Inner Stretch','Head to Head Comparison','Oligo','Learn Cat Beard','Leg bending side sitting','Left swag','Left Shoulder Stretch','Left humeral triple extension','Left Back Stretch','Left Neck Stretch','Left Front Thigh Stretch','Left single swing arm','Left calf stretch','Left Bow Step Twist Stretch','Left Bend','Raise hand with left hand','Left hand raised','Left hand circle','Cover the top of the head with the left hand','Left Oblique Pull Down Half Squat','Left Waist Stretch','Left leg pressed backwards','Left iliopsoas muscle stretching','with phone case','Rolling on Bed','Opening and closing the door','Opening and closing steps','jumping jack','Open Hands and Cross Legs','Bend down and walk','Bare Hand Squat','Bare Hand Hard Pull Boat Action','Wandering and Pacing','Wear a mask','Wear a hat','Wear glasses','Fan','Gesture Compared to Heart','Touch Shoulder','Wrist band','Arm Cross','Arm Circle','Tie Hair','Strike Ten Step Fist','Snap Fingers','Sneezing','Playing Tai Chi','Making a phone call','Playing Long Fist','Throw Throw Pillow','Throw Bottle','Throw paper plane','Chest Expansion','Sweeping the floor','Twist waist','Arm Press Down','Pulse Detection','Grab Hair','Scratch your ears and cheeks','Shooting action','Surrender','Shake Legs','Shake clothes','Raise Hands and Support Face','Head hugging','Wipe the Neck','Smoking','Pull Bag Zipper','Pull the curtains','Take a Card','Taking Scenery','Weeding','Floor mopping','Twist Cup Cover','Arched Hand Gift','Collate Building Blocks','Press the temple wheel to scrape the eye socket','Sidebag','Scratching','Block the Sun','Wave','Fist','Cover Ears','Hammering the Table','Beat Back','Chest Beating','Beating Legs','Pull Out Ears','Pinch waist','Pushing the wall to accelerate running','Push Pull Luggage','Cart','Shoulder Lift','Knee Lift','Heel lifting','Handshake','clench your fist and start running','Rubbing Hands for Heating','Moving a chair','Wave Hands and Lift Legs','Shake your head','Shake Tree','Flower Shaker','Paper Throwing Cannon','Remove glasses','Touch chin','Touch the back of the head','Touch Ear Clip Back','Touch the neck and tilt the head','Touch waist and clip back','Touch the forehead','Touch the Nose','Umbrella Support','Tearing Paper','Wipe Cup','Clean the Table','Glasses wiping','Hip Up Kick Jump','Salute','Knock Calculator','Keyboard tapping','Knock on the door','Mummy Jump','Comb Hair','Chair Split Leg Press','Chair frame left leg press','Skew Head Biye','Twist your neck and touch your head','Bi Xin','Dishwashing','Wash Face','Washing Vegetables','Washing clothes','Measure distance','Applying Lipstick','Deep Squat','Slide Jump Shot','Nod','Like','Warm Up Relaxation Jump','Guessing Fists','Play Double Step','Playing the Spacewalk','Playing Windmills','Paper stapling with stapler','Hands off','Draw','Eye Care Exercise','Hard Pull Swing','Wearing clothes','Wearing shoes','Standing Long Jump','Vertical Middle Finger','Standing Right Leg Lift','Standing Right Rear Leg Lift','Standing Left Leg Lift','Standing Left Rear Leg Lift','Standing Elbow Knee Rotation','Standing Jump Transformation','Standing Twist','Standing Hip Circle','Standing Touch Toe','Simplified Tai Chi Cloud Hand','Simplified Tai Chi Inverted Arm','Simplified Tai Chi Twin Peaks Through Ears','Simplified Tai Chi Right Kick','Simplified Tai Chi as if sealed off','Simplified Tai Chi Left Right Shuttle','Simplified Tai Chi Tower Knee Depression Step','Simplified Tai Chi Sea Bottom Needle','Simplified Tai Chi White Crane Spreads Wings','Simplified Tai Chi Turn Left Kick','Simplified Tai Chi Turning Move Block Hammer','Simplified Tai Chi Wild Horse Split Mane','Arrow Squat','Arrow Squat Kick','Arrow Squat Knee Lift','Arrow Squat Jump','Tie Rope','Combination Recommendation','Cross legged Penguin','Cross your legs and cover your eyes','Flip Book','Draw Circle at Elbow','Shoulder Wrap','Chest Stretch','Knee joint surround','Hip Kick Jump','selfie','Kangaroo Shake','Take a step forward','Lie on the table','Running','Stamping feet','Kicking','Kicking Trees','Jump','Transfer pen','Continuous Right Knee Strike','Continuous Left Knee Strike','Cover the Eyes and Lift the Legs','Arnold Recommendation','Bow','Air Kiss','Riding a bicycle','High Leg Lift','Duck Step','Applause','Bulging cheeks','Mobile Card 1','Mobile Card 2','Mobile Card 3']

attr_words = ['A','2','3','4','5','6','7','8','9','10','J','Q','K','JOKER','Alligator','Antelope','Anthozoan','Bald Eagle','Bat','Beetle','Big Spirobranchus Giganteus','Bombing Variegata','Butterfly','Camel','Chameleon','Crab','Cyclophiops major','Dolphin 001','Elephant','Gharial','Giraffe','Hedgehog','Hippopotamus','Hummingbird','Jellyfish','Koala','Leopard','Lizard','Mantis','Monkey','Mudskipper','Nautilus','Ostrich','Otter','Owl','Panda','Parrot','Peacock Mantis Shrimp','Penguin','Phycodurus eques','Poison dart frog','Protozoon','Red panda','Scorpion','Sea hare','Sea turtle','Snailfish','Spider','Spongia','Squirrel','Starfish','Synchiropus Splendidus','Tiger','Tuatara','Weaverbird','Whale','White bear','White shark','Antennarius','Balistidae','Box fish','Butterflyfishes','Caranx sexfasciatus','Clean shrimps','Clownfish','Cuttlefish','Dolphin 002','Gold arowana','Green Mandarin','Grouper','Hippocampus','Jellyfish','Leafy Sea Dragous','Limulus','Manta Ray','Mouse Spot','Muraenesox','Murex Troscheli','Nautilus','Octopus','Paddlefish','Palinuridae','Porcupine Fishes','Powder blue tang','Pterophyllum','Rhinecanthus Aculeatus','Sea Anemone','Sea Cucumber','Sea Hare','Sea Lion','Sea Turtle','Sea Urchin','Seal','Shark','Sickle Fish','Snailfish','Snake Eel','Starfish','Stonefish','Trumpet fish','Tunny','Walrus','Whale','Wolf eel',]

def get_png_file_paths(directory):
    file_paths = []
    for file_name in os.listdir(directory):
        if file_name.endswith(".png"):
            file_path = os.path.join(directory, file_name)
            file_paths.append(file_path)
            print(("processing", file_name))
    return file_paths

def generate_imgs(path,track_name):#形成一个字典，key=track_name，value=imgs_path
    imgs_path={}
    for i in track_name:# track_name  ['0001C2T0070', '0005C2T0075', '0005C5T0146', '0005C5T0261', '0022C4T0015', '0065C3T0009']
        tracklet_path=path+'/'+str(i)+'/'      # tracklet_path '/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/lidong/VTF_PAR-main/dataset/MARS/pad_mars_dataset/0065C3T0009/'
        result=get_png_file_paths(tracklet_path) 
        imgs_path[i]=result     # imgs_path {'0001C2T0070': ['/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/lidong/VTF_PAR-main/dataset/MARS/pad_mars_dataset/0001C2T0070/0001C2T0070F014.jpg',
    #breakpoint()
    return imgs_path

def generate_event(path,track_name):#形成一个字典，key=track_name，value=imgs_path
    imgs_path={}
    for i in track_name:# track_name  ['0001C2T0070', '0005C2T0075', '0005C5T0146', '0005C5T0261', '0022C4T0015', '0065C3T0009']
        tracklet_path=path+'/'+str(i)+'/'      # tracklet_path '/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/lidong/VTF_PAR-main/dataset/MARS/pad_mars_dataset/0065C3T0009/'
        result=get_png_file_paths(tracklet_path) 
        imgs_path[i]=result     # imgs_path {'0001C2T0070': ['/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/lidong/VTF_PAR-main/dataset/MARS/pad_mars_dataset/0001C2T0070/0001C2T0070F014.jpg',
    #breakpoint()
    return imgs_path

def make_dir(path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)

def get_label_embeds(labels):
    model = SentenceTransformer('all-mpnet-base-v2')
    embeddings = model.encode(labels)
    return embeddings


#生成一个标签的字典，其中键代表tracklets_id,值为一个list（为样本的标签值）
def generate_label(filename):
    result_dict = {}
    with open(filename, 'r') as file:
        lines = file.readlines()
        
        for line in lines:
            line = line.strip()
            if line:
                parts = line.split(" ")
                imgname_str = line.split("/")[1].split(" ")[0]
                key = parts[0] + "/" + imgname_str + "_aps"
                value = np.zeros(114)
                for part in parts[2:]:
                    num = int(part)
                    value[num] = 1
                result_dict[key] = value
    #breakpoint()
    return result_dict

def generate_event_label(filename):
    result_dict = {}
    with open(filename, 'r') as file:
        lines = file.readlines()
        
        for line in lines:
            line = line.strip()
            if line:
                parts = line.split(" ")
                imgname_str = line.split("/")[1].split(" ")[0]
                key = parts[0] + "/" + imgname_str + "_dvs"
                value = np.zeros(114)
                for part in parts[2:]:
                    num = int(part)
                    value[num] = 1
                result_dict[key] = value
    
    return result_dict

def generate_data_description(save_dir):

    dataset = EasyDict()
    dataset.description = 'mars'
    dataset.root=os.path.join(r'/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/dataset/Poker_Event/Poker_rgb_event')
    dataset.attr_name=attr_words
    dataset.words=np.array(attr_words)
    dataset.attr_vectors=get_label_embeds(attr_words)
    result_dict=generate_label("/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/lidong/VTB-main/dataset/poker/poker.txt")
    result_event_dict= generate_event_label("/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/lidong/VTB-main/dataset/poker/poker.txt")
    trainval_name=[]
    trainval_event_name=[]
    test_name=[]
    test_event_name=[]
    trainval_gt_list=[]
    test_gt_list=[]
    track_name=[]
    track_event_name=[]
    track_gt_list=[]
    

    #track_name_file=open("/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/lidong/VTB-main/dataset/poker/poker.txt",'r',encoding='utf8').readlines()
    with open("/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/lidong/VTB-main/dataset/poker/poker.txt", "r") as file:
        lines = file.readlines()
    for name in lines :
        curLine = name.split(" ")[0]
        imgname_str = curLine.split("/")[1]
        track_name.append(curLine + "/" + imgname_str + "_aps")  #track_name ['0001C2T0070', '0005C2T0075', '0005C5T0146', '0005C5T0261', '0022C4T0015', '0065C3T0009']
    
    #event
    with open("/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/lidong/VTB-main/dataset/poker/poker.txt", "r") as file:
        lines = file.readlines()
    for name in lines :
        curLine = name.split(" ")[0]
        imgname_str = curLine.split("/")[1]
        track_event_name.append(curLine + "/" + imgname_str + "_dvs")


    #trainval_name_file=open("/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/dataset/Poker_Event/Poker_Event_list/train_label.txt",'r',encoding='utf8').readlines()
    with open("/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/dataset/Poker_Event/Poker_Event_list/train_label.txt", "r") as file:
        lines = file.readlines()
    for name in lines :
        curLine = name.split(" ")[0]
        imgname_str = curLine.split("/")[1]
        trainval_name.append(curLine + "/" + imgname_str + "_aps")  #['0001C2T0070', '0005C2T0075', '0005C5T0146']



    with open("/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/dataset/Poker_Event/Poker_Event_list/train_label.txt", "r") as file:
        lines = file.readlines()
    for name in lines :
        curLine = name.split(" ")[0]
        imgname_str = curLine.split("/")[1]
        trainval_event_name.append(curLine + "/" + imgname_str + "_dvs")



    #test_name_file=open("/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/dataset/Poker_Event/Poker_Event_list/test_label.txt",'r',encoding='utf8').readlines()
    with open("/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/dataset/Poker_Event/Poker_Event_list/test_label.txt", "r") as file:
        lines = file.readlines()
    for name in lines :
        curLine = name.split(" ")[0]
        imgname_str = curLine.split("/")[1]
        test_name.append(curLine + "/" + imgname_str + "_aps")  #['0005C5T0261', '0022C4T0015', '0065C3T0009']
    
    with open("/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/dataset/Poker_Event/Poker_Event_list/test_label.txt", "r") as file:
        lines = file.readlines()
    for name in lines :
        curLine = name.split(" ")[0]
        imgname_str = curLine.split("/")[1]
        test_event_name.append(curLine + "/" + imgname_str + "_dvs")

    #breakpoint()
    for gt in track_name:
        curLine=gt.split(" ")[0]
           
        track_gt_list.append(result_dict[curLine])   #[array([1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0,       0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0]), array([1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0,       0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0]), array([1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0,       0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0]), array([1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0,       0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0]), array([1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0,       0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0]), array([1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0,       0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0])]
    
    for gt in trainval_name:
        curLine=gt.split(" ")[0]
        
        trainval_gt_list.append(result_dict[curLine]) #训练真值，同上

    for gt in test_name:
        
        curLine=gt.split(" ")[0]
        test_gt_list.append(result_dict[curLine]) #测试真值

    #dataset.track_name=track_name
    dataset.test_name=test_name#4908   #['0005C5T0261', '0022C4T0015', '0065C3T0009']
    dataset.test_event_name=test_event_name
    dataset.trainval_name=trainval_name#11452  ['0001C2T0070', '0005C2T0075', '0005C5T0146']
    dataset.trainval_event_name = trainval_event_name

    dataset.track_name=dataset.trainval_name+dataset.test_name
    dataset.track_event_name = dataset.trainval_event_name+dataset.test_event_name

    dataset.trainval_gt_list=trainval_gt_list
    dataset.test_gt_list=test_gt_list

    dataset.track_gt_list=track_gt_list  #数组真值
    dataset.result_dict = result_dict    #字典形式真值
    dataset.result_event_dict = result_event_dict
    dataset.attr_vectors = get_label_embeds(attr_words)
    
    dataset.label = np.concatenate((np.array(trainval_gt_list),np.array(test_gt_list)), axis=0)  #将数组合在一起变成一个总的数组，加了一层[]
    #assert dataset.label.shape == (64522+32386, 300)
    assert dataset.label.shape == (16216+8199, 114)

    dataset.partition = EasyDict()
    dataset.attr_name = attr_words
    #dataset.partition.test = np.arange(64522, 64522+32386)  
    dataset.partition.test = np.arange(16216, 16216+8199)
    #dataset.partition.trainval = np.arange(0, 64522)  
    dataset.partition.trainval = np.arange(0, 16216)
    #dataset.weight_trainval = np.mean(dataset.label[dataset.partition.trainval], axis=0).astype(np.float32)   
    # 包含每个tracklet中图片的地址
    path1="/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/dataset/Poker_Event/Poker_rgb_event"
    dataset.track_imgs_path=generate_imgs(path1,track_name)
    dataset.track_event_path=generate_imgs(path1,track_event_name)
    breakpoint()
    with open(os.path.join(save_dir, 'poker.pkl'), 'wb+') as f:
        pickle.dump(dataset, f)

if __name__ == "__main__":
    save_dir = '/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/lidong/VTF_PAR-main/dataset/MARS/'
    generate_data_description(save_dir)
