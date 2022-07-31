# Uncle_Tony_assignment

Uncle Tony is head of the admissions office at a famous university.
Every year, many new students enroll at the school and must be divided into different classes.
The principle behind the division has always been clear: maximize the classroom heterogeneity, to mix students with different peculiarities.
However, professors have complained that dividing the students by their preparation would be better for the teaching.
Uncle Tony knows that such a division could raise protests from the parents, but agrees with the teachers, and would like to find a compromise.
However, his office is busy with other tasks; they don’t have enough time to organize the split according to new procedures, before the beginning of the new year.

## Part 1
Could you please help Uncle Tony? He would like you to build a simple software (that can be delivered in the form a Python Notebook) that will help him do the job.
Uncle Tony sent you a .csv with data registered for every student of the previous year before the enrollment.
Specifically:
    - Every row refers to a single student
    - The last column contains the results of a skills test done to check the student preparation
    - Every other column contains the answers to generic questions about the background of the student.

Uncle Tony also have attached a few questions to the .csv, to synthesize the requests:
1. Can you show me what kind of students have enrolled in the previous years, and what was their preparation?
2. Can you build an algorithm to separate the students in a number of classes between 2 and 10, trying to keep similar results of the skill test, but different answers for the other questions? Can you describe how you choose that specific number of classes?
3. Uncle Tony would like to understand if there is a relation between the background answers and the test result. Could you help him to do so, investigating if there are particular correlations between background answers and the skills test?
 
## Part 2
Uncle Tony is satisfied with your idea and decided to apply your algorithm to create the classrooms.
However, it wants to convince the board of the school that this approach should become the new standard. Can you please prepare ONE slide, to help presents the benefits of using your algorithm?

## Part 3
Uncle Tony has one last request: can you structure your code in an object-oriented paradigm using
(simple) objects and classes, also testing your results with dedicated unit tests? Feel free to use any Python editor to create your new script!

## How to use main.py

List of arguments to launch main.py from command line:

--path_data -> String containing the path relative to the students' form data. 
--pca_dimension -> Optional, default=20. Number of principal components to use for classes split. 
--visualize -> Optional, default=False. True to visualize the analysis relative to the different algorithm steps.
--num_clusters -> Optional, default=13. Number of clusters to use for classes split.
--output_path -> Optional. Path in which you want to save the df with students splitted in classes.


Example of a working command:

python3 main.py --path_data '~/Downloads/data_challenge_v2.csv' --visualize True --pca_dimension 2 --num_clusters 7 --output_path '/set/your/output/path'
