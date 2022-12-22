function [iTRAXX_Tranches, iTRAXX_Tranches_Values, iTRAXX_Tranches_Year] =...
          load_iTRAXX_Tranches(iTRAXX_Path,iTRAXX_Date,ext)
%%LOAD_ITRAXX_Tranches loads values of tranches of iTRAXX. The
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
%   load_iTRAXX_Tranches(iTRAXX_Path,iTRAXX_Date): ext will be wildcard
%   load_iTRAXX_Tranches(iTRAXX_Path,iTRAXX_Date,ext): file extension as
%                                                      char
%
% See also: readtable
    
    % Check arguments
    arguments
        iTRAXX_Path (1,:) char
        iTRAXX_Date (1,:) char 
        ext (1,:) char = '*' 
    end

    
    % List of files
    files = dir([iTRAXX_Path,'/iTRAXX_Tranches_',iTRAXX_Date,'.',ext]);
    if length(files)>1
        error('File is not unique.')
    end

    % Read table
    temp=readtable([files.folder,'/',files.name]);

    % Remove columns with missing values
    temp(:,any(ismissing(temp),1))=[];

    % Extract data
    tranches=temp{2:end,1}; % tranche names
    iTRAXX_Tranches_Values=temp{2:end,2:end}; % tranche values
    iTRAXX_Tranches_Year=temp{1,2:end}; % tranche years

    tranches = tranches(2:end);
    iTRAXX_Tranches=zeros(length(tranches),2);

    for i=1:1:length(tranches)
        tranche = tranches(i);
        tmp = erase(tranche{1},[' ','%']);
        points = strsplit(tmp,'-');
        points = arrayfun(@(p) str2double(p{1}),points);
        if strcmp(tranche{1}(end),'%')
            points = points/100;
        end
        iTRAXX_Tranches(i,:)=points;
    end
end