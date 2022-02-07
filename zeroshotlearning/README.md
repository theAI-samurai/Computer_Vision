# zero-shot-learning
Implementation of Zero-Shot Learning algorithm  
  
Zero-Shot learning method aims to solve a task without receiving any example of that task at training phase.  
It simply allows us to recognize objects *we have not seen before*.   
  
### Classes  
**Train Classes:**  
arm, boy, bread, chicken, child, computer, ear, house, leg, sandwich, television, truck, vehicle, watch, woman  
**Zero-Shot Classes:**  
car, food, hand, man, neck  
  
## Usage  
$**python3**  detect_object.py  input-image-path  
  
### Example  
$**cd**  src  
$**python3**  detect_object.py  ../test.jpg  
**->** --- Top-5 Prediction ---  
**->** 1- vehicle  
**->** 2- truck  
**->** 3- car  
**->** 4- house  
**->** 5- chicken  
  
  
*Test image is a beautiful green Jaguar E-Type.*  
*All related prediction results are ranked in first three.*  