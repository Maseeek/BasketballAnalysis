Basketball Analysis
# Description
## Most Basic Outline
A program which takes in a video and determines the proportion of shots made and the release angle of each shot.
## A more detailed analysis
This program uses computer vision to determine the trajectory of a ball, determines the predicted trajectory of the ball and records whether the shot is a make and the release angle. 
### Ball detection
A key concept needed to make the program work effectively is the detection of the ball.

This program uses a collection of techniques to effectively detect the ball:

- Initially a frame is taken, from the video, and it is transformed from BGR (standard colour) to GRAY (grayscale) so that objects that are of multiple colours are not detected as separate objects but detected by their edges.
- Then the frame is blurred using a gaussian blur so that only outlines of larger objects are detectable to eliminate small objects or camera grain to be detected. 
- Then the Hough circle function is used to detect any circles in the frame so that any objects which may be the ball will be detected.
- Finally, we iterate through the circles detected and determine the closest circle to the position of the previous circle recorded so that we can keep detecting the same ball and not detect another circular object which is in an abnormal position in relation to the last position the ball was detected.


### Trajectory
One of the most visual components of the program is the ball trajectory and a popular feature is the trajectory prediction.
#### *Trajectory Recording*
The ball is detected using the above-described ball detection features which are encapsulated in the function findBall() which returns the chosen circle which is determined to be the basketball: consisting of the ball position and radius.
#### *Trajectory Prediction*
*What is polynomial regression –* polynomial regression is a technique used to find the coefficients of a polynomial function so that a set of positions can be modelled by a polynomial function.

Ball prediction is a key feature of the program. It is achieved by feeding the past positions of the ball and using polynomial regression to determine an equation to model the trajectory of the ball path. 
### Recording Field Goal Percentage and Release Angle 
#### *Field Goal Percent*
All shots are recorded in an array, with a 1 representing a ‘make’ and a 0 representing a ‘miss’. We also record FGA (Field Goal Attempts) and FGM (Field Goals Made), using a basic calculation of FGM/FGA \* 100, we can calculate the FG% (Field Goal Percentage) which we will display to the user.
#### *Release Angle*
We calculate release angle on a shot-by-shot basis and when the video terminates this allows us to calculate an average release angle and we can segment this by makes and misses, allowing us to see correlation between release angle and field goal percentage.
### Modelling results
Through the recording of all the shots, their outcomes and release angles, we can model the shots and see how certain statistics can affect the outcome of the shot. We model the relationship between the shot outcome and release angle, which allows us to determine the optimal shot angle.
