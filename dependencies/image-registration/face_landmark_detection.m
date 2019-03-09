function face_landmark_detection(folder_path, wildcard)

    python = '<UPDATE ME>'; % Full path to the python executable
    dlib = '<UPDATE ME>'; % Full path to dlib's 'shape_predictor_68_face_landmarks.dat'
    command_str = [python ' ' ...
                   fullfile(pwd, 'dependencies/image-registration/face_landmark_detection.py ') ...
                   dlib ' '...
                   '%s "%s"'];
    command_str = sprintf(command_str, folder_path, wildcard);
    system(command_str);

end