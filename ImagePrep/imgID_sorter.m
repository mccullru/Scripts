clc; clear

%{

Takes the sheets of recorded SuperDove and Sentinel-2 images with the 
image ID, rating, descriptor, and best images and creates new columns on 
the same sheet that include totals for the ratings and descriptions

%}

folderPath = "B:\Thesis Project\Raw Imagery\ImageIDs\Individual_AOI_Lists\Sentinel2";

files = dir(fullfile(folderPath, "*.xlsx"));

for f = 1:length(files)
    
    filename = files(f).name;
    fullpath = fullfile(folderPath, filename);
    fprintf('Processing File: %s\n', filename)

    AOI = readtable(fullpath);
    
    count_1 = sum(AOI.Rating == 1);
    count_2 = sum(AOI.Rating == 2);
    count_3 = sum(AOI.Rating == 3);
    rating_counts = [count_1, count_2, count_3];
    
    total = size(AOI, 1);
    per1 = (count_1/total)*100;
    per2 = (count_2/total)*100;
    per3 = (count_3/total)*100;
    rating_pers = [per1, per2, per3];
    
    counts_1rat = zeros(1, 9);
    counts_2rat = zeros(1, 9);
    counts_3rat = zeros(1, 9);
    
    for i = 1:3
    
        count_pc = sum(~cellfun('isempty', regexp(AOI.Descriptor(AOI.Rating ==i), '(^|,)\s*pc\s*(,|$)', 'once')));
        count_c = sum(~cellfun('isempty', regexp(AOI.Descriptor(AOI.Rating ==i), '(^|,)\s*c\s*(,|$)', 'once')));
        count_cs = sum(~cellfun('isempty', regexp(AOI.Descriptor(AOI.Rating ==i), '(^|,)\s*cs\s*(,|$)', 'once')));
        count_t = sum(~cellfun('isempty', regexp(AOI.Descriptor(AOI.Rating ==i), '(^|,)\s*t\s*(,|$)', 'once')));
        count_pia = sum(~cellfun('isempty', regexp(AOI.Descriptor(AOI.Rating ==i), '(^|,)\s*pia\s*(,|$)', 'once')));
        count_ia = sum(~cellfun('isempty', regexp(AOI.Descriptor(AOI.Rating ==i), '(^|,)\s*ia\s*(,|$)', 'once')));
        count_m = sum(~cellfun('isempty', regexp(AOI.Descriptor(AOI.Rating ==i), '(^|,)\s*m\s*(,|$)', 'once')));
        count_b = sum(~cellfun('isempty', regexp(AOI.Descriptor(AOI.Rating ==i), '(^|,)\s*b\s*(,|$)', 'once')));
        count_a = sum(~cellfun('isempty', regexp(AOI.Descriptor(AOI.Rating ==i), '(^|,)\s*a\s*(,|$)', 'once')));
    
        if i==1
            counts_1rat = [count_pc, count_c, count_cs, count_t, count_pia, count_ia, count_m, count_b, count_a];
        elseif i==2
            counts_2rat = [count_pc, count_c, count_cs, count_t, count_pia, count_ia, count_m, count_b, count_a];
        elseif i==3
            counts_3rat = [count_pc, count_c, count_cs, count_t, count_pia, count_ia, count_m, count_b, count_a];
        end
    end
    
    all_counts = [counts_1rat; counts_2rat; counts_3rat];
    
    
    AOI.best = strings(size(AOI, 1), 1);
    
    for i = 1:height(AOI)
        if AOI.Rating(i) == 3
            AOI.best(i) = AOI.IDs(i);
        end
    end
    
    % Add additional column information to original collection sheet
    add_columns = ["final_ratings", "rating_count", "rating_percent", "pc_count", "c_count", "cs_count", "t_count", "pia_count", "ia_count", "m_count", "b_count", "a_count"];
    
    for i = 1:length(add_columns)
        AOI.(add_columns(i)) = nan(height(AOI),1);
    end
    
    %%% Fill out columns %%%
    
    % rating labels
    AOI.final_ratings(1:3) = [1;2;3];     
    AOI.final_ratings(4) = 123;

    % rating counts
    for i = 1:3
        AOI.rating_count(i) = rating_counts(i);
    end
    AOI.rating_count(4) = rating_counts(1) + rating_counts(2) + rating_counts(3);

    % rating percentages
    for i = 1:3
        AOI.rating_percent(i) = rating_pers(i);
    end
    
    % Descriptor counts
    AOI.pc_count(1:3) = all_counts(:,1);
    AOI.c_count(1:3) = all_counts(:,2);
    AOI.cs_count(1:3) = all_counts(:,3); 
    AOI.t_count(1:3) = all_counts(:,4);
    AOI.pia_count(1:3) = all_counts(:,5);
    AOI.ia_count(1:3) = all_counts(:,6);
    AOI.m_count(1:3) = all_counts(:,7);
    AOI.b_count(1:3) = all_counts(:,8);
    AOI.a_count(1:3) = all_counts(:,9);

    AOI.pc_count(4) = all_counts(1,1) + all_counts(2,1) + all_counts(3,1);
    AOI.c_count(4) = all_counts(1,2) + all_counts(2,2) + all_counts(3,2);
    AOI.cs_count(4) = all_counts(1,3) + all_counts(2,3) + all_counts(3,3);
    AOI.t_count(4) = all_counts(1,4) + all_counts(2,4) + all_counts(3,4);
    AOI.pia_count(4) = all_counts(1,5) + all_counts(2,5) + all_counts(3,5);
    AOI.ia_count(4) = all_counts(1,6) + all_counts(2,6) + all_counts(3,6);
    AOI.m_count(4) = all_counts(1,7) + all_counts(2,7) + all_counts(3,7);
    AOI.b_count(4) = all_counts(1,8) + all_counts(2,8) + all_counts(3,8);
    AOI.a_count(4) = all_counts(1,9) + all_counts(2,9) + all_counts(3,9);
    
    writetable(AOI, fullpath)

    fprintf('Finished Processing: %s\n', filename);

end

fprintf('All files finished')


















