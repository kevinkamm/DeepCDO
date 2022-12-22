function iTRAXX_CDS = load_iTRAXX_CDS(iTRAXX_Path,iTRAXX_Date,ext)
%%LOAD_ITRAXX_CDS loads values of CDS of the constituents of iTRAXX. The
% data may be any format usable by *readtable*, e.g., csv, xlsx,...
%
% Input:
%   iTRAXX_Path (1 x d1 char): contains the path to data
%   iTRAXX_Date (1 x d2 char): contains the date of data with format
%                               dd_mm_yyyy
%   ext (1 x d3 char, optional): extension of data file
%
% Output:
%   iTRAXX_Tranches (d x 1 cell): tranche names
%   iTRAXX_Tranches_Values (d x n double): tranche values
%   iTRAXX_Tranches_Year (1 x n double): tranche years
%
% Usage: 
%   load_iTRAXX_CDS(iTRAXX_Path,iTRAXX_Date): ext will be wildcard
%   load_iTRAXX_CDS(iTRAXX_Path,iTRAXX_Date,ext): file extension as char
%
% See also: readtable
    
    % Check arguments
    arguments
        iTRAXX_Path (1,:) char
        iTRAXX_Date (1,:) char 
        ext (1,:) char = '*' 
    end

    
    % List of files
    files = dir([iTRAXX_Path,'/iTRAXX_CDS_',iTRAXX_Date,'.',ext]);
    if length(files)>1
        error('File is not unique. Check for different extensions')
    end

    % Read table
    temp=readtable([files.folder,'/',files.name]);

    % Remove columns with missing values
    temp(:,any(ismissing(temp),1))=[];

    % Extract data
%     iTRAXX_Tranches=temp{2:end,1}; % tranche names
%     iTRAXX_Tranches_Values=temp{2:end,2:end}; % tranche values
%     iTRAXX_Tranches_Year=temp{1,2:end}; % tranche years
    iTRAXX_CDS = temp;

end