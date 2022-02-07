# Character Recognition from Car Licence Plate - India

This is an Hybrid implementation of Image processing and Deep Learning

1. Character are segmented from the Plate First, using Image processing
2. This segmented cahracters are then fed to the Network for Prediction of characters

This method was prefered as time for annotation of data was limited for us (at the time of writing) and 
Mobile SSD network does give a bounding box of characters, hence we segmented the characters first and then got inference on the characters

