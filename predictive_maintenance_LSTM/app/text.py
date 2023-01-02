import gtts  
from playsound import playsound  

t1 = gtts.gTTS("class 2: engine maintenance urgent")  
t1.save("class2.mp3")   
t1 = gtts.gTTS("class 1: engine maintenance necessary")
t1.save("class1.mp3")
t1 = gtts.gTTS("schedule engine maintenance")  
t1.save("reg60.mp3")
playsound("class1.mp3")
playsound("class2.mp3")
playsound("reg60.mp3")