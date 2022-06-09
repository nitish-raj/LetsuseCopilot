
-- Select all from students table
SELECT * FROM students;

-- select top performing students based on avergae marks per semester

SELECT * FROM students WHERE marks > (SELECT AVG(marks) FROM students);



-- Code does following:
-- 1. Select all from students table
SELECT * FROM students;



-- 2. categorise students based on their marks
SELECT * FROM students WHERE marks > (SELECT AVG(marks) FROM students);


-- 3. group students based on class and section
SELECT * FROM students GROUP BY class, section;


-- 4. select top performing students based on avergae marks per semester




