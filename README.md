Download Link: https://assignmentchef.com/product/solved-cs433-exercise-7-support-vector-machine-svm-using-sgd-and-coordinate-descent
<br>
Goals.       The goal of this exercise is to

<ul>

 <li>Implement and debug Support Vector Machine (SVM) using SGD and coordinate descent.</li>

 <li>Derive updates for the coordinate descent algorithm for the dual optimization problem for SVM.</li>

 <li>Implement and debug the coordinate descent algorithm.</li>

 <li>Compare it to the primal solution.</li>

</ul>

Setup, data and sample code.               Obtain the folder labs/ex07 of the course github repository

<a href="https://github.com/epfml/ML_course/tree/master/labs/ex07">github.com/epfml/ML</a> <a href="https://github.com/epfml/ML_course/tree/master/labs/ex07">course</a>

We will finally depart from using the height-weight dataset and instead use the larger CERN dataset from Project 1 in this exercise. We have provided sample code templates that already contain useful snippets of code required for this exercise.

<h1>1           Support Vector Machines using SGD</h1>

Until now we have implemented linear and logistic regression to do classification. In this exercise we will use the Support Vector Machine (SVM) for classification. As we have seen in the lecture notes, the original optimization problem for the Support Vector Machine (SVM) is given by

<em>N</em>

<em>λ         </em><sub>2</sub>

min                                      <em> w                                                                           </em>(1)

<em>w</em>

where <em>` </em>: R → R, <em>`</em>(<em>z</em>) := max{0<em>,</em>1 − <em>z</em>} is the <em>hinge loss </em>function. Here for any <em>n</em>, 1 ≤ <em>n </em>≤ <em>N</em>, the vector <em>x</em><em><sub>n </sub></em>∈ R<em><sup>D </sup></em>is the <em>n<sup>th </sup></em>data example, and <em>y<sub>n </sub></em>∈ {±1} is the corresponding label.

Problem 1 (SGD for SVM):

Implement stochastic gradient descent (SGD) for the original SVM formulation (1). That is in every iteration, pick one data example <em>n </em>∈ [<em>N</em>] uniformly at random, and perform an update on <em>w </em>based on the (sub-)gradient of the <em>n<sup>th </sup></em>summand of the objective (1). Then iterate by picking the next <em>n</em>.

<ol>

 <li>Fill in the notebook functions calculate accuracy(y, X, w) which computes the accuracy on the training/test dataset for any <em>w </em>and calculate primal objective(y, X, w, lambda ) which computes the total primal objective (1).</li>

 <li>Derive the SGD updates for the original SVM formulation and fill in the notebook function calculate stochastic gradient() which should return the stochastic gradient of the total cost function (loss plus regularizer) with respect to <em>w</em>. Finally, use sgd for svm demo() provided in the template for training.</li>

</ol>

<h1>2           Support Vector Machines using Coordinate Descent</h1>

As seen in class, another approach to train SVMs is by considering the dual optimization problem given by

<em>YXX</em><sup>&gt;</sup><em>Yα </em>such that                 0 ≤ <em>α<sub>n </sub></em>≤ 1 ∀<em>n                                          </em>(2)

where <em>Y </em>:= diag(<em>y</em>), and <em>X </em>∈ R<em><sup>N</sup></em><sup>×<em>D </em></sup>again collects all <em>N </em>data examples as its rows, as usual. In this approach we optimize over the dual variables <em>α </em>and map the solutions back to the primal vector <em>w</em>.

Problem 2 (Coordinate Descent for SVM):

Derive the coordinate descent algorithm updates for the dual (2) of the SVM formulation. That is, in every iteration, pick a coordinate <em>n </em>∈ [<em>N</em>] uniformly at random, and fully optimize the objective (2) with respect to that coordinate alone.

After updating that coordinate <em>α<sub>n</sub></em>, update the corresponding primal vector <em>w </em>such that the first-order correspondence is maintained, that is that always <em>w </em>= <em>w</em>(<em>α</em>) := <em><sub>λ</sub></em><u><sup>1</sup></u><em>X</em><sup>&gt;</sup><em>Yα</em>. Then iterate by picking the next coordinate <em>n</em>.

<ol>

 <li>Mathematically derive the coordinate update for one coordinate <em>n </em>(finding the closed-form solution to maximization over just that coordinate), when given <em>α </em>and corresponding <em>w</em>.</li>

 <li>Fill in the notebook functions calculate coordinate update() which should compute the coordinate update for a single desired coordinate and calculate dual objective() which should return the objective (loss) for the dual problem (2) .</li>

 <li>Finally train your model using coordinate descent (here ascent) using the given function sgd for svm demo() in the template. Compare to your SGD implementation. Which one is faster? (Compare the training objective values (1) for the <em>w </em>iterates you obtain from each method).</li>

</ol>

<h1>Theory Excercises</h1>

Problem 3 (Kernels):

In class we have seen that many kernel functions <em>k</em>(<em>x</em><em>,x</em><sup>0</sup>) can be written as inner products <em>φ</em>(<em>x</em>)<sup>&gt;</sup><em>φ</em>(<em>x</em><sup>0</sup>) for a suitably chosen feature map <em>φ</em>(·). Let us say that such a kernel function is <em>valid</em>. We further discussed many operations on valid kernel functions that result again in valid kernel functions. Here are two more.

<ol>

 <li>Let <em>k</em><sub>1</sub>(<em>x</em><em>,x</em><sup>0</sup>) be a valid kernel function. Let <em>f </em>be a polynomial with positive coefficients. Show that is a valid kernel.</li>

 <li>Show that <em>k</em>(<em>x</em><em>,x</em><sup>0</sup>) = exp(<em>k</em><sub>1</sub>(<em>x</em><em>,x</em><sup>0</sup>) is a valid kernel assuming that <em>k</em><sub>1</sub>(<em>x</em><em>,x</em><sup>0</sup>) is a valid kernel. HINT: You are allowed to take limits.</li>

</ol>