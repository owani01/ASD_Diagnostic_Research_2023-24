# ASD_Diagnostic_Research_2023-24

Autism Spectrum Disorder (ASD) is a neurodevelopmental disorder which affects around 1 in 36 children in America. 
As of October 2023, despite the amount of groundbreaking research identifying autism’s neurobiological underpinnings, 
there is yet to be an official diagnostic method for the condition; most diagnoses are made through developmental assessments 
made by physicians and developmental specialists, making it prone to human subjectivity. Hence, this research attempted to create 
an accurate diagnostic model for ASD based on fMRI (functional magnetic resonance imaging) and EEG (electroencephalogram) scans 
by utilizing machine learning algorithms. The procedure for creating such a methodology involves many preliminary steps pertaining 
to the collection and cleaning of data for the model, including (1) the collection of datasets containing fMRI and EEG scans of autistic 
and neurotypical individuals, (2) the preprocessing and cleaning of the data using MATLAB, Python libraries, and other neuroimaging software, 
and (3) extracting only relevant features from the preprocessed data using specialized feature extraction methods. 
The diagnostic model can then be developed by picking specific machine/deep learning algorithms and related libraries to 
optimize the model’s parameters, then splitting the preprocessed data into subsections for training (‘feeding’ data to algorithms) 
and testing the model (testing whether the model’s prediction based on data input matches the actual diagnosis). The model’s 
effectiveness in diagnosis will be determined using the calculated evaluation metrics (including accuracy scores, sensitivity, 
specificity, precision, and F1 score) from the test sample. The methodology will be repeated until its diagnostic accuracy is 
optimized, after which the model will be analyzed to identify patterns contributing to accurate diagnosis, and then it’ll be 
compared to existing diagnostic methods for ASD (including behavioral assessments and other diagnostics using fMRI/EEG data) 
to understand the model’s effectiveness in the grand scheme of autism’s various diagnostic methods. The research will then 
address future paths of research pertaining in this field for next steps, as well as proposing the potential of machine learning 
for other aspects of autism as well. 
