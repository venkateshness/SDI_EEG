for CURSUBJ in $(cat $1)
do
# Mac users, an app Cyberduck comes in very handy listing the files from the AWS bucket on-the-go

## if "aws: not found" appears, you might need to do PATH=~/.local/bin:$PATH

#Video 1
    aws s3 cp s3://fcp-indi/data/Projects/HBN/EEG/${CURSUBJ}/EEG/preprocessed/csv_format/Video3_data.csv src_data/video1/${CURSUBJ}/ --no-sign-request
    aws s3 cp s3://fcp-indi/data/Projects/HBN/EEG/${CURSUBJ}/EEG/preprocessed/csv_format/Video3_event.csv src_data/video1/${CURSUBJ}/ --no-sign-request
    aws s3 cp s3://fcp-indi/data/Projects/HBN/EEG/${CURSUBJ}/EEG/preprocessed/csv_format/Video3_chanlocs.csv src_data/video1/${CURSUBJ}/ --no-sign-request

#Restingstate
    aws s3 cp s3://fcp-indi/data/Projects/HBN/EEG/${CURSUBJ}/EEG/preprocessed/csv_format/RestingState_data.csv src_data/rest/${CURSUBJ}/ --no-sign-request
    aws s3 cp s3://fcp-indi/data/Projects/HBN/EEG/${CURSUBJ}/EEG/preprocessed/csv_format/RestingState_event.csv src_data/rest/${CURSUBJ}/ --no-sign-request
    aws s3 cp s3://fcp-indi/data/Projects/HBN/EEG/${CURSUBJ}/EEG/preprocessed/csv_format/RestingState_chanlocs.csv src_data/rest/${CURSUBJ}/ --no-sign-request

#Video 2    
    aws s3 cp s3://fcp-indi/data/Projects/HBN/EEG/${CURSUBJ}/EEG/preprocessed/csv_format/Video2_data.csv src_data/video2/${CURSUBJ}/ --no-sign-request
    aws s3 cp s3://fcp-indi/data/Projects/HBN/EEG/${CURSUBJ}/EEG/preprocessed/csv_format/Video2_event.csv src_data/video2/${CURSUBJ}/ --no-sign-request
    aws s3 cp s3://fcp-indi/data/Projects/HBN/EEG/${CURSUBJ}/EEG/preprocessed/csv_format/Video2_chanlocs.csv src_data/video2/${CURSUBJ}/ --no-sign-request

done
