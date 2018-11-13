# 00-Intro to Machine Learning

# SYLLABUS

**Title:** Machine Learning in Art, Sound, and Language
**Course:** DSI 450
**Date:** Tue, Nov 6, 13, 20, 6:00 - 9:00pm
**Instructor:** Mike Heavers
**Contact:** mheavers@gmail.com


## Summary

Machine Learning is rapidly being implemented in all facets of life to parse data, learn from that data, and draw conclusions based on what it has learned. Inherent in this is the fact that machines see the world very differently than we do.  In assigning creative tasks to machines such as drawing pictures, making music, and creating written language, we expose these differences in strange and unexpected ways. 

This class will look at how to leverage various Machine Learning frameworks to generate art, music, and written language that belongs not fully to the human world or the computational one.

Machine Learning is not for the faint of heart - it can often involve a number of dependencies to install, models can be slow to train and processor intensive, and results unpredictable. But in that unpredictability lies the creativity! 

While we we will be examining some of the most beginner friendly machine learning tools, you will be expected to install some software, look at some code, and run some computations outside of class. But don’t fear, we will arrive at some interesting results together regardless of whether you are an absolute beginner or have done some coding / machine learning in the past.


## Introductions

What is your name?
Why are you here?


## Schedule

**Nov 6, 2018**

Introduction to Machine Learning


- Neural Networks
- Optimization
- Practical Applications
- History of Machine Learning
- Modern Day Machine Learning Artists

Machine Learning for Images


- Convolutional Neural Networks
- How Machines See Images
- Style Transfer

**Nov 13, 2018**

Machine Learning in Language


- Recurrent Neural Networks
- Word Vectors
- Long Short Term Memory
- Predictive Keyboards

**Nov 20, 2018**

Machine Learning in Sound and Music


- Speech Synthesis
- Speech Mimicry
- Pitch Detection
- Music Mimicry

Recap
Course Evaluation
Next Steps


# INTRODUCTION


## WHAT IS MACHINE LEARNING?

Machine Learning is the concept of assigning an objective to a computer, giving it a concept of success, failure, and progress[**](https://blog.openai.com/reinforcement-learning-with-prediction-based-rewards/), and giving it the ability to evaluate and adapt along the way to find an optimal solution. This is done through **neural networks**. 

They process information similar to the way the brain does, through [**Neurons**](https://www.quora.com/What-are-neurons-in-machine-learning). Neurons receive input, perform some sort of processing on that input, and send it off as output (like a filter).

Essentially, that output can only be a **number** or a **label**. But in our models and frameworks, we can turn those numbers and labels into something much, much more meaningful.

The magic of machine learning is in the fact that we can have thousands of them working together in unison to accomplish a task, intuitively reacting to each other, firing in response to each other just like the brain. **Cells that fire together wire together.**

**Is It Art?**

Clearly some think so. Paris Collective [Obvious](http://obvious-art.com/)’s *Portrait of Edmond Bellamy* ******is set to become [the first AI generated artwork sold at auction at Christie’s](https://www.christies.com/features/A-collaboration-between-two-artists-one-human-one-a-machine-9332-1.aspx).


![](https://www.christies.com/media-library/images/features/articles/2018/08/09/a-collaboration-between-two-artists-one-human-one-a-machine/edmond-de-belamy-framed-cropped.jpg?w=780)


And Mario Klingemann won the [lumen prize](https://lumenprize.com/) in technology-driven art for teaching an AI to recognize and render the human form from a stick figure.


![Mario Klingemann’s Butcher’s Son](https://d2mxuefqeaa7sj.cloudfront.net/s_3AD63810368EE33E6E6A21A2675FCFE6E5680F16543413CBB619F3A55FF07EBD_1539218271892_butchers-son.jpg)




## HISTORY

**Early 1900s**
[Ada Lovelace](https://en.wikipedia.org/wiki/Ada_Lovelace)(first computer programmer) aspires to build a "calculus of the nervous system"
[Charles Babbage](https://en.wikipedia.org/wiki/Charles_Babbage) (Ada’s mentor) - invents the first mechanical computer

**1940s**
Neural nets pioneered in the 1940s with [Turing B-Type Machines](https://en.wikipedia.org/wiki/Unorganized_machine) - organized logic gates responding to each other. Based on research being done into **neuroplasticity** - the idea that through repeat activations of neighboring neurons, efficiencies are gained, forming the basis for unsupervised learning. 

**1950s**
[Marvin Minsky](https://en.wikipedia.org/wiki/Marvin_Minsky)'s [SNARC](https://en.wikipedia.org/wiki/Stochastic_neural_analog_reinforcement_calculator) is the first Neural Network Machine.

**1990s**
Until this point, the primary application is in **support vector machines**, essentially calculating probability through **linear regression**. Examples include image classification and text categorization.


- IBM's Deep Blue beat Gary Kasparov in chess.
- [LSTMs](https://en.wikipedia.org/wiki/Long_short-term_memory) invented, improving efficiencies.
- [MNIST](https://en.wikipedia.org/wiki/MNIST_database) database for handwriting recognition by American Census Bureau and some high school kids (free labor!). MNIST is sort of the “hello world” of Machine Learning.


![MNIST](https://cdn-images-1.medium.com/max/1200/1*yBdJCRwIJGoM7pwU-LNW6Q.png)



**2009**

- "deep belief networks" were introduce for speech recognition.
- [ImageNet](https://en.wikipedia.org/wiki/ImageNet) data set Introduced. Catalyst for our current AI boom.

**2011**
IBM's [Watson](https://en.wikipedia.org/wiki/Watson_(computer)) beats human competitors in Jeopardy.

**2012**
C**onvolutional Neural Networks**(CNNs) - Huge advancement for image processing!

**2014**

- [DeepFace](https://en.wikipedia.org/wiki/DeepFace) by Facebook increases facial recognition accuracy by 27% over previous systems, rivaling human performance.
- [Deep Dream](https://en.wikipedia.org/wiki/DeepDream) - brings AI to the attention of artists and tech generalists.
![Google Deep Dream](https://upload.wikimedia.org/wikipedia/commons/6/65/Aurelia-aurita-3-0049.jpg)


**TODAY**
Deep Learning is more accessible than ever. Computation can be done in the cloud, we have more and more data available from which to learn from, and frameworks for machine learning are increasingly numerous. Popular frameworks such as [Tensorflow](https://en.wikipedia.org/wiki/TensorFlow) have been extrapolated to more and more accessible languages, including [Keras](https://keras.io/) and [ML5](https://ml5js.org/), which we'll be using in this class.


## WHAT CAN WE DO WITH MACHINE LEARNING?

In theory, everything the human brain can / could do, and then some! Right now common applications include:


- Prediction - The core goal of machine learning. Comes in two basic forms:
  - Classification - assigning a label
  - Regression - assigning a quantity


  This is essentially all a neuron can do!
  
- [Sentiment analysis](https://www.analyticsvidhya.com/blog/2018/07/hands-on-sentiment-analysis-dataset-python/) - determining whether words as a corpus are positive, negative, etc.


![](https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2018/07/wordcloud_11.png)




- [Pose detection](https://storage.googleapis.com/tfjs-models/demos/posenet/camera.html) 


https://www.youtube.com/watch?v=PCBTZh41Ris&


[https://youtu.be/PCBTZh41Ris](https://youtu.be/PCBTZh41Ris)


![We use a pose detector P : Y ′ → X ′ to obtain pose joints for the source person that are transformed by our normalization process Norm into joints for the target person for which pose stick figures are created. Then we apply the trained mapping G (source)](https://d2mxuefqeaa7sj.cloudfront.net/s_B596028CC42A3B3D0806231BF508AB1EBAC4EFEFDDE0B3A78DEC3C9488383D81_1540496549542_Screen+Shot+2018-10-25+at+12.41.34+PM.png)




- [Image segmentation](https://sthalles.github.io/deep_segmentation_network/)
![Image Segmentation](https://cdn-images-1.medium.com/max/2000/1*MQCvfEbbA44fiZk5GoDvhA.png)




- [Image colorization](http://richzhang.github.io/colorization/) (take a black and white photo, make it color)
- [Resolution](https://letsenhance.io/) (Enhance!) 


![Enhancing a photo with Enhancenet-PAT](https://petapixel.com/assets/uploads/2017/11/enhancefeat-800x420.jpg)




https://www.youtube.com/watch?v=LhF_56SxrGk&


[https://youtu.be/LhF_56SxrGk](https://youtu.be/LhF_56SxrGk)



- [Text to image](http://t2i.cvalenzuelab.com/) - Type some text, the neural network constructs what it thinks that text might look like in image form. 


![Text to Image](https://d2mxuefqeaa7sj.cloudfront.net/s_3AD63810368EE33E6E6A21A2675FCFE6E5680F16543413CBB619F3A55FF07EBD_1538894412767_Screen+Shot+2018-10-07+at+3.39.51+PM.png)

- [Image to text](http://www.cs.toronto.edu/~nitish/nips2014demo/index.html) - A machine captions what it sees in a photo using natural language
- [Speech synthesis](https://lyrebird.ai/) (generating voice w text) 
- [Word vectors](https://medium.com/@jayeshbahire/introduction-to-word-vectors-ea1d4e4b84bf) - define the relationship between words. 
- [Translation](https://translate.google.com/) - improving word-by-word translation by better identifying common word combinations across languages.

Questions So Far?


## THE BASICS OF ML

**What are Neural Networks?**

Artificial Neural Networks (ANN) are a collection of connected units or nodes called [artificial neurons](https://en.wikipedia.org/wiki/Artificial_neuron) which loosely model the [neurons](https://en.wikipedia.org/wiki/Neuron) in a biological [brain](https://en.wikipedia.org/wiki/Brain).

Each connection, like the [synapses](https://en.wikipedia.org/wiki/Synapse) in a biological [brain](https://en.wikipedia.org/wiki/Brain), can transmit a signal from one artificial neuron to another. An artificial neuron that receives a signal can process it and then signal additional artificial neurons connected to it.

In common ANN implementations, the signal at a connection between artificial neurons is a number, and the output of each artificial neuron is computed by some **non-linear** function of the sum of its inputs. 

 Typically, artificial neurons are aggregated into layers. Different layers may perform different kinds of transformations on their inputs.
 
 **How do we evaluate the findings of a single neuron?** 

Examine **Linear regression** as a simple example: finding a line of best fit. Our evaluation criteria is called a **loss function**, determining distance between line and data points and penalizing us for greater distances. But doing this across all neurons would be incredibly slow. It would mean having to evaluate every single datapoint!

![Linear Regression and Mean-Squared Error](https://ml4a.github.io/images/figures/lin_reg_error.png)


**So how Do We Train Neural Networks?**

The most common evaluation of progress across a neural network as a whole is called **Gradient descent**. It’s sort of like finding your way down a mountain in the dark with a weak flashlight. You look in all directions, see which step takes you furthest down, take a step, then repeat.

We are looking for one thing: the **weights** of a certain variable or variables that minimizes the **cost.**


![Calculating Gradient Descent](https://ml4a.github.io/images/figures/lin_reg_mse_gradientdescent.png)


**What is Cost?**

A cost function is a measure of [how wrong the model is in terms of its ability to estimate the relationship between X and Y](https://towardsdatascience.com/machine-learning-fundamentals-via-linear-regression-41a5d11f5220).


![Gradient Descent](https://ml4a.github.io/images/figures/opt2a.gif)


**Gradient Descent IRL**

Neural networks for practical problems are rarely linear! They have many dimensions / variables. Rather than having a nice easy bowl shape, they're full of hills and troughs, which can greatly complicate things.


![Dimensionality](https://ml4a.github.io/images/figures/non_convex_function.png)


The more variables we have to solve for, the more neurons we need, increasing processing time exponentially. With just brute force guessing, trying to classify a set of digits from 0-9 would take 10*12000 guesses (784 input neurons, 15 hidden, 10 output) . There are only 10*80 atoms in the universe! So we need to optimize.

**Optimization**

One way to do this is to figure out how far we step in a given direction. This is influenced by learning rate. Too far, we may end up back uphill, too close, we'll never get down.

There's gonna be some redundancy in our progress - if we have a long stretch of downhill, we don't need to evaluate every single step. In ML we can batch it out.

When we have high confidence or continued success in our path, we can use momentum to keep going in more or less the same path. If we keep choosing uphill, we eventually run out of momentum and need to find a better path.

Another optimization method is [back propagation](https://en.wikipedia.org/wiki/Backpropagation) - taking what we've learned at our current progress step and applying it backward to all previous ones to figure out what is no longer necessary. This allows us to increase our learning rate. 

As we train our model, we might notice patterns, called **features.** If we were analyzing hand writing, those features might ultimately be loops, ascenders, descenders, etc. If we were identifying cats, we might look for eyes, ears, noses, tails. 

But features exist in different layers, and the initial layers might just be edges, patterns, colors. Only as these layers build on each other do they take the shape of what we're looking for. 

Identifying features allows us to vastly speed up and increase the accuracy of our models.

**How Do We Evaluate the Network As a Whole?**

We could run our model over our full dataset, but we are in danger of **overfitting** - presuming the findings from our data is representative of all data in existence. This is a hard to avoid problem of all sciences. 

To evaluate, we can split the data into **training and test sets**. This allows us to make assumptions based on our learnings, and then test them to find out if they hold up.

**How Do We Know How Much to Train?**

Consider studying for a test. You improve by studying, but at some point all that studying doesn’t actually improve results / retention, or rather it's not worth the cost. We can’t remember everything.

To figure out how much to study, you can add a **validation set** in which you examine all your assumptions. 



## 
## ML ARTISTS

[**Mario Klingemann**](http://quasimondo.com/)
Excellent Work in Style Transfer


https://twitter.com/quasimondo/status/1043458889969221632



[**Gene Kogan**](https://vimeo.com/139123754)
I got a lot of the class material from him. Artist and instructor at ITP


https://twitter.com/genekogan/status/857922705412239362



[**Botnik Studios**](http://botnik.org/apps/writer/)
Great tools for AI-driven NLP


![Botnik Machine Learning Generated Coachella Poster](http://botnik.org/content/coachella.jpg)



[**AI Weirdness**](https://vimeo.com/139123754)
Awesome examples of AI gone weird. E.G:

*Neural Network College Course List*

*General Almosts of Anthropology*
*Deathchip Study*
*Advanced Smiling Equations*
*Genies and Engineering*
*Practicum Geology-Love*
*Electronics of Faces*
*Devilogy*
*Psychology of Pictures in Archaeology*
*Melodic Studies in Collegine Mathematics*
*Advanced Computational Collegy*
*The Papering II*
*Professional Professional Pattering II*
*Introduction to Economic Projects and Advanced Care and Station Amazies*
*Every Methods*
*Chemistry of Chemistry*
*Internship to the Great*
*The Sciences of Prettyniss*
*Geophing and Braining*
*Survivery*

[**Jun Yan Zhu**](http://people.csail.mit.edu/junyanz/)
Most ML advancements seem to start at the research level. [This guy](https://github.com/junyanz) has been responsible for so many advancements, including CycleGAN and Vid2Vid, which we’ll look at soon.


![Vid2Vid by Jun Yan Zhu](https://d2mxuefqeaa7sj.cloudfront.net/s_B596028CC42A3B3D0806231BF508AB1EBAC4EFEFDDE0B3A78DEC3C9488383D81_1540596527063_vid2vid.gif)


[**Sofia Crespo**](https://www.instagram.com/soficrespo91/?hl=en)


![](https://pbs.twimg.com/media/Dj1v5cGX0AYZSIL.jpg)




**So What Machine Learning Architecture Should I Use?**

There are a bunch, and it depends on what you are doing. Here is a (vastly oversimplified) summary of a few key networks:

**Convolutional Neural Networks (CNNs) -** In these networks, neurons only concern themselves with data of neighboring cells, and are very efficient at simplifying information down to only its essential bits, and filtering out noise. Therefore, for things that require a lot of data, like images, sound, and video.

**Recurrent Neural Networks (RNNs)** - These networks have a concept of memory - propagating their findings backward through the network. These networks are good at prediction - finding the next thing in a sequence, and adapting to learnings as that sequence evolves - such as with sequences of words, arrangements of like objects, etc.

**Deep Belief Networks** - The back-propagation of recurrent networks comes at a cost - information must be labeled, which is inefficient, as much data can actually be represented numerically. In order to account for this, deep belief networks perform unsupervised learning: making decent decisions about their locally relevant data, that are perhaps not optimal for the network as a whole. If you’ve seen Deep Dream Puppy Slugs, you can see understand how these networks might continue to see one kind of a thing in an image, rather than looking at the image as a whole. Deep Dream has been programmed to be hyper-aware to local neurons, instructed to find the features it is in charge of wherever possible.

**Generative Adversarial Networks (GANs)** - Two networks working together by working against each other. The first network generates content, and the second judges the outcome. Does the generated content look natural, or artificial? The generator tries to fool the discriminator, and the discriminator tries to catch the generator in the act of forgery. 


# MACHINE LEARNING WORKSHOPS

We will be looking at Machine Learning in 3 different core areas. Click the links to go to the course material for each specific section.

[**Course Files**](https://drive.google.com/open?id=1xn3ocvqpSPbdl_n32hBvM8RdH--JRGhA)
This Google Drive contains all of the applications and source code we will be using in class. Please download the zip files in advance of each course.

[**Machine Learning In Art**](https://paper.dropbox.com/doc/Machine-Learning-for-Art-cB3DnphC884fRYCuMl8Oy)
ML as it relates to image processing, video, and other visual mediums

[**Machine Learning in Language**](https://paper.dropbox.com/doc/Machine-Learning-in-Language-QnAPvbpQIHkABOR88NoCq)
ML with respect to text, speech, and language processing

[**Machine Learning in Sound+Music**](https://paper.dropbox.com/doc/Machine-Learning-in-SoundMusic-9fOw3k0F22ysJnNDlCw9Q)
ML in audible contexts: voice, music, sound, etc.

# WHERE TO NEXT?

This course barely scratches the surface of what is possible with Machine Learning. Here are some good resources to take your explorations further.

## Courses

[Machine Learning for Musicians and Artists](https://www.kadenze.com/courses/machine-learning-for-musicians-and-artists/info) - Haven’t taken it, heard it’s good
**Instructor:** [Rebecca Fiebrink](https://www.doc.gold.ac.uk/~mas01rf/homepage/)
**Source:** [Kadenze](https://www.kadenze.com/)

[The Neural Aesthetic](http://ml4a.github.io/classes/itp-F18/) - the course from which much of this material was drawn
**Instructor:** [Gene Kogan](http://genekogan.com/)
**Source:** [ITP at NYU](https://itp.nyu.edu)

[Fast AI](http://www.fast.ai/) - A great library as well as courses to get you started in a framework that prioritizes quick algorithms.

[Google Colab](https://colab.research.google.com/notebooks/welcome.ipynb) - Collection of ML-related Jupyter Notebooks to work through including a Machine Learning crash course and intro to Pandas. 
22

## Tools

[Lobe](https://lobe.ai/) **-** Visual AI model training tools

[RunwayML](https://runwayapp.ai/) **-** Suite of plugins, models, and tools for more advanced ML applications from fellow ITP alums.

[ML4A-OFX](https://github.com/ml4a/ml4a-ofx) - Tools and code samples / applications using [openFrameworks](https://openframeworks.cc)

[Fast A](http://www.fast.ai/)I - A tool and course for AI using Python/[PyTorch](https://pytorch.org/)

[Pix2Pix Unity](https://github.com/keijiro/Pix2Pix) - Style Transfer within a Game Engine. [openCV for Unity](https://assetstore.unity.com/packages/tools/integration/opencv-for-unity-21088) also has some ML models incorporated, including [YOLO](https://pjreddie.com/darknet/yolo/)

[Puppeteer](https://pptr.dev/) - A headless web browser that can be used in conjunction with the language of your choice ([node.js](https://www.scrapehero.com/how-to-build-a-web-scraper-using-puppeteer-and-node-js/)?) for scraping data.


## Inspiration

[Google Chrome AI Experiments](https://experiments.withgoogle.com/collection/ai)


## Community

[ML4A Slack](https://ml-4a.slack.com) - Started at ITP, open to anyone


## Data

[Andy Warhol Photography Archive](http://cantorcollection.stanford.edu/IT_267?sid=2966&x=29353&display=thu&x=29354)

[Internet Archive Book Archive](https://www.flickr.com/photos/internetarchivebookimages/) 

[List of Datasets on Wikipedia](https://en.wikipedia.org/wiki/List_of_datasets_for_machine_learning_research) - Generally searches for “datasets for machine learning” on Google will yield a wealth of public data sources.

[Data Scraping](https://www.coursera.org/courses?query=web%20scraping) - A list of courses / tools for data scraping from Coursera. Arguably more important than machine learning knowledge is access to data, so this is worth learning.

[Beautiful Soup](https://pypi.org/project/beautifulsoup4/) - The Data Scraper I’ve Used (Python)


## Course Evaluation

I want to make this course as useful and fun as possible, and I want to become a better teacher. Your input helps make this possible, and helps Make + Think + Code program as a whole. Please fill out [this evaluation](https://goo.gl/forms/il2vQqAdZ6kQx2SW2) form before you leave.

