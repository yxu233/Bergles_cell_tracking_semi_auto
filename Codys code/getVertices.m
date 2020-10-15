function [dimJ,centerJ,verticesJ] = getVertices(name)
if contains(name,'211')
    dimJ = [1077 1064 205];
    centerJ = dimJ./2;
    verticesJ = [0 0 18; dimJ(1) 0 35; ...
        0 dimJ(2) 2; dimJ(1) dimJ(2) 21; ...
        0 0 118; dimJ(1) 0 135; ...
        0 dimJ(2) 102; dimJ(1) dimJ(2) 121];
    verticesJ = [verticesJ(:,1:2) verticesJ(:,3)-6]; % for fine tuning
elseif contains(name,'216')
    dimJ = [1247 1097 198];
    centerJ = dimJ./2;
    verticesJ = [0 0 2; dimJ(1) 0 17; ...
        0 dimJ(2) 2; dimJ(1) dimJ(2) 23; ...
        0 0 102; dimJ(1) 0 117; ...
        0 dimJ(2) 102; dimJ(1) dimJ(2) 123];
    verticesJ = [verticesJ(:,1:2) verticesJ(:,3)-0]; % for fine tuning
elseif contains(name,'204')
    dimJ = [1051 1052 205];
    centerJ = dimJ./2;
    verticesJ = [0 0 14; dimJ(1) 0 15; ...
        0 dimJ(2) 0; dimJ(1) dimJ(2) 4; ...
        0 0 114; dimJ(1) 0 115; ...
        0 dimJ(2) 100; dimJ(1) dimJ(2) 104];
    verticesJ = [verticesJ(:,1:2) verticesJ(:,3)-2]; % for fine tuning
elseif contains(name,'260')
    dimJ = [1072 1043 200];
    centerJ = dimJ./2;
    verticesJ = [0 0 18; dimJ(1) 0 34; ...
        0 dimJ(2) 13; dimJ(1) dimJ(2) 24; ...
        0 0 118; dimJ(1) 0 134; ...
        0 dimJ(2) 113; dimJ(1) dimJ(2) 124];
    verticesJ = [verticesJ(:,1:2) verticesJ(:,3)-7]; % for fine tuning
    
    
    %% control:
elseif contains(name,'056')
    dimJ = [1245 1220 188];
    centerJ = dimJ./2;
    verticesJ = [0 0 10; dimJ(1) 0 12; ...
        0 dimJ(2) 4; dimJ(1) dimJ(2) 8; ...
        0 0 107; dimJ(1) 0 112; ...
        0 dimJ(2) 104; dimJ(1) dimJ(2) 108];
    verticesJ = [verticesJ(:,1:2) verticesJ(:,3)+3]; % for fine tuning
    
elseif contains(name,'264')
    dimJ = [1088 1105 220];
    centerJ = dimJ./2;
    verticesJ = [0 0 5; dimJ(1) 0 48; ...
        0 dimJ(2) 0; dimJ(1) dimJ(2) 32; ...
        0 0 105; dimJ(1) 0 148; ...
        0 dimJ(2) 100; dimJ(1) dimJ(2) 132];
    verticesJ = [verticesJ(:,1:2) verticesJ(:,3)+6]; % for fine tuning
    
elseif contains(name,'089')
    dimJ = [1052 1046 188];
    centerJ = dimJ./2;
    verticesJ = [0 0 2; dimJ(1) 0 2; ...
        0 dimJ(2) 2; dimJ(1) dimJ(2) 2; ...
        0 0 102; dimJ(1) 0 102; ...
        0 dimJ(2) 102; dimJ(1) dimJ(2) 102];
    verticesJ = [verticesJ(:,1:2) verticesJ(:,3)+3]; % for fine tuning
elseif contains(name,'115')
    dimJ = [1072 1086 194];
    centerJ = dimJ./2;
    verticesJ = [0 0 11; dimJ(1) 0 7; ...
        0 dimJ(2) 2; dimJ(1) dimJ(2) 6; ...
        0 0 111; dimJ(1) 0 107; ...
        0 dimJ(2) 102; dimJ(1) dimJ(2) 106];
    verticesJ = [verticesJ(:,1:2) verticesJ(:,3)-0]; % for fine tuning
elseif contains(name,'186')
    dimJ = [1042 1053 189];
    centerJ = dimJ./2;
    verticesJ = [0 0 5; dimJ(1) 0 5; ...
        0 dimJ(2) 5; dimJ(1) dimJ(2) 5; ...
        0 0 105; dimJ(1) 0 105; ...
        0 dimJ(2) 105; dimJ(1) dimJ(2) 105];
    verticesJ = [verticesJ(:,1:2) verticesJ(:,3)+3]; % for fine tuning
    
    
elseif contains(name,'385')
    dimJ = [1087 1071 196];
    centerJ = dimJ./2;
    verticesJ = [0 0 4; dimJ(1) 0 11; ...
        0 dimJ(2) 4; dimJ(1) dimJ(2) 11; ...
        0 0 104; dimJ(1) 0 111; ...
        0 dimJ(2) 104; dimJ(1) dimJ(2) 111];
    verticesJ = [verticesJ(:,1:2) verticesJ(:,3)-0]; % for fine tuning
elseif contains(name,'420')
    dimJ = [1109 1053 190];
    centerJ = dimJ./2;
    verticesJ = [0 0 4; dimJ(1) 0 6; ...
        0 dimJ(2) 4; dimJ(1) dimJ(2) 6; ...
        0 0 104; dimJ(1) 0 106; ...
        0 dimJ(2) 104; dimJ(1) dimJ(2) 106];
    verticesJ = [verticesJ(:,1:2) verticesJ(:,3)+2]; % for fine tuning
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
elseif contains(name,'277')
    dimJ = [1082 1072 210];
    centerJ = dimJ./2;
    verticesJ = [0 0 13; dimJ(1) 0 64; ...
        0 dimJ(2) 3; dimJ(1) dimJ(2) 34; ...
        0 0 113; dimJ(1) 0 164; ...
        0 dimJ(2) 103; dimJ(1) dimJ(2) 134];
    verticesJ = [verticesJ(:,1:2) verticesJ(:,3)-5]; % for fine tuning
elseif contains(name,'235')
    dimJ = [1074 1052 197];
    centerJ = dimJ./2;
    verticesJ = [0 0 1; dimJ(1) 0 27; ...
        0 dimJ(2) 1; dimJ(1) dimJ(2) 31; ...
        0 0 101; dimJ(1) 0 127; ...
        0 dimJ(2) 101; dimJ(1) dimJ(2) 131];
    verticesJ = [verticesJ(:,1:2) verticesJ(:,3)-2]; % for fine tuning
    %% 20x images
elseif contains(name,'001')
    dimJ = [1054 1060 165];
    centerJ = dimJ./2;
    verticesJ = [0 0 18; dimJ(1) 0 18; ...
        0 dimJ(2) 4; dimJ(1) dimJ(2) 7; ...
        0 0 118; dimJ(1) 0 118; ...
        0 dimJ(2) 104; dimJ(1) dimJ(2) 107];
    verticesJ = [verticesJ(:,1:2) verticesJ(:,3)-0]; % for fine tuning
elseif contains(name,'030')
    dimJ = [1050 1059 201];
    centerJ = dimJ./2;
    verticesJ = [0 0 16; dimJ(1) 0 30; ...
        0 dimJ(2) 6; dimJ(1) dimJ(2) 26; ...
        0 0 116; dimJ(1) 0 130; ...
        0 dimJ(2) 106; dimJ(1) dimJ(2) 126];
    verticesJ = [verticesJ(:,1:2) verticesJ(:,3)+2]; % for fine tuning
elseif contains(name,'033')
    dimJ = [1038 1058 192];
    centerJ = dimJ./2;
    verticesJ = [0 0 12; dimJ(1) 0 21; ...
        0 dimJ(2) 5; dimJ(1) dimJ(2) 16; ...
        0 0 112; dimJ(1) 0 121; ...
        0 dimJ(2) 105; dimJ(1) dimJ(2) 116];
    verticesJ = [verticesJ(:,1:2) verticesJ(:,3)-0]; % for fine tuning
elseif contains(name,'037')
    dimJ = [1081 1081 198];
    centerJ = dimJ./2;
    verticesJ = [0 0 18; dimJ(1) 0 18; ...
        0 dimJ(2) 2; dimJ(1) dimJ(2) 7; ...
        0 0 118; dimJ(1) 0 118; ...
        0 dimJ(2) 102; dimJ(1) dimJ(2) 107];
    verticesJ = [verticesJ(:,1:2) verticesJ(:,3)-0]; % for fine tuning
elseif contains(name,'090')
    dimJ = [1076 1050 195];
    centerJ = dimJ./2;
    verticesJ = [0 0 14; dimJ(1) 0 21; ...
        0 dimJ(2) 1; dimJ(1) dimJ(2) 8; ...
        0 0 114; dimJ(1) 0 121; ...
        0 dimJ(2) 101; dimJ(1) dimJ(2) 108];
    verticesJ = [verticesJ(:,1:2) verticesJ(:,3)-2]; % for fine tuning
elseif contains(name,'097')
    dimJ = [1071 1056 195];
    centerJ = dimJ./2;
    verticesJ = [0 0 10; dimJ(1) 0 19; ...
        0 dimJ(2) 2; dimJ(1) dimJ(2) 10; ...
        0 0 110; dimJ(1) 0 119; ...
        0 dimJ(2) 102; dimJ(1) dimJ(2) 110];
    verticesJ = [verticesJ(:,1:2) verticesJ(:,3)+1]; % for fine tuning
elseif contains(name,'099')
    dimJ = [1050 1059 201];
    centerJ = dimJ./2;
    verticesJ = [0 0 0; dimJ(1) 0 15; ...
        0 dimJ(2) 0; dimJ(1) dimJ(2) 15; ...
        0 0 102; dimJ(1) 0 115; ...
        0 dimJ(2) 102; dimJ(1) dimJ(2) 115];
    verticesJ = [verticesJ(:,1:2) verticesJ(:,3)+3]; % for fine tuning
    
    
    
    
    
    
    
    
    
elseif contains(name,'339')
    dimJ = [1076 1066 191];
    centerJ = dimJ./2;
    verticesJ = [0 0 5; dimJ(1) 0 17; ...
        0 dimJ(2) 6; dimJ(1) dimJ(2) 22; ...
        0 0 105; dimJ(1) 0 117; ...
        0 dimJ(2) 106; dimJ(1) dimJ(2) 122];
    verticesJ = [verticesJ(:,1:2) verticesJ(:,3)-3]; % for fine tuning
elseif contains(name,'369')
    dimJ = [1064 1053 192];
    centerJ = dimJ./2;
    verticesJ = [0 0 4; dimJ(1) 0 43; ...
        0 dimJ(2) 4; dimJ(1) dimJ(2) 43; ...
        0 0 104; dimJ(1) 0 143; ...
        0 dimJ(2) 104; dimJ(1) dimJ(2) 143];
    verticesJ = [verticesJ(:,1:2) verticesJ(:,3)-4]; % for fine tuning
    
    
    
    
    
    
elseif contains(name,'470')
    dimJ = [1047 1040 160];
    centerJ = dimJ./2;
    verticesJ = [0 0 10; dimJ(1) 0 14; ...
        0 dimJ(2) 9; dimJ(1) dimJ(2) 21; ...
        0 0 110; dimJ(1) 0 114; ...
        0 dimJ(2) 109; dimJ(1) dimJ(2) 121];
    verticesJ = [verticesJ(:,1:2) verticesJ(:,3)+0]; % for fine tuning
elseif contains(name,'471')
    dimJ = [1024 1024 151];
    centerJ = dimJ./2;
    verticesJ = [0 0 10; dimJ(1) 0 11; ...
        0 dimJ(2) 1; dimJ(1) dimJ(2) 18; ...
        0 0 110; dimJ(1) 0 111; ...
        0 dimJ(2) 101; dimJ(1) dimJ(2) 118];
    verticesJ = [verticesJ(:,1:2) verticesJ(:,3)-1]; % for fine tuning
elseif contains(name,'500')
    dimJ = [1029 1040 254];
    centerJ = dimJ./2;
    verticesJ = [0 0 8; dimJ(1) 0 75; ...
        0 dimJ(2) 8; dimJ(1) dimJ(2) 58; ...
        0 0 108; dimJ(1) 0 175; ...
        0 dimJ(2) 108; dimJ(1) dimJ(2) 158];
    verticesJ = [verticesJ(:,1:2) verticesJ(:,3)-8]; % for fine tuning
elseif contains(name,'510')
    dimJ = [1189 1228 194];
    centerJ = dimJ./2;
    verticesJ = [0 0 3; dimJ(1) 0 5; ...
        0 dimJ(2) 16; dimJ(1) dimJ(2) 13; ...
        0 0 103; dimJ(1) 0 105; ...
        0 dimJ(2) 116; dimJ(1) dimJ(2) 113];
    verticesJ = [verticesJ(:,1:2) verticesJ(:,3)-3]; % for fine tuning
elseif contains(name,'520')
    dimJ = [1038 1043 192];
    centerJ = dimJ./2;
    verticesJ = [0 0 1; dimJ(1) 0 1; ...
        0 dimJ(2) 15; dimJ(1) dimJ(2) 19; ...
        0 0 101; dimJ(1) 0 101; ...
        0 dimJ(2) 115; dimJ(1) dimJ(2) 119];
    verticesJ = [verticesJ(:,1:2) verticesJ(:,3)-1]; % for fine tuning
elseif contains(name,'530')
    dimJ = [1059 1064 208];
    centerJ = dimJ./2;
    verticesJ = [0 0 25; dimJ(1) 0 3; ...
        0 dimJ(2) 50; dimJ(1) dimJ(2) 25; ...
        0 0 125; dimJ(1) 0 103; ...
        0 dimJ(2) 150; dimJ(1) dimJ(2) 125];
    verticesJ = [verticesJ(:,1:2) verticesJ(:,3)-5]; % for fine tuning
elseif contains(name,'540')
    dimJ = [1074 1071 197];
    centerJ = dimJ./2;
    verticesJ = [0 0 2; dimJ(1) 0 2; ...
        0 dimJ(2) 33; dimJ(1) dimJ(2) 33; ...
        0 0 102; dimJ(1) 0 102; ...
        0 dimJ(2) 133; dimJ(1) dimJ(2) 133];
    verticesJ = [verticesJ(:,1:2) verticesJ(:,3)-3]; % for fine tuning
elseif contains(name,'550')
    dimJ = [1044 1041 195];
    centerJ = dimJ./2;
    verticesJ = [0 0 0; dimJ(1) 0 0; ...
        0 dimJ(2) 40; dimJ(1) dimJ(2) 30; ...
        0 0 100; dimJ(1) 0 100; ...
        0 dimJ(2) 140; dimJ(1) dimJ(2) 130];
    verticesJ = [verticesJ(:,1:2) verticesJ(:,3)-4]; % for fine tuning
elseif contains(name,'490')
    dimJ = [1083 1069 178];
    centerJ = dimJ./2;
    verticesJ = [0 0 3; dimJ(1) 0 8; ...
        0 dimJ(2) 3; dimJ(1) dimJ(2) 12; ...
        0 0 103; dimJ(1) 0 108; ...
        0 dimJ(2) 103; dimJ(1) dimJ(2) 112];
    verticesJ = [verticesJ(:,1:2) verticesJ(:,3)-2]; % for fine tuning
elseif contains(name,'491')
    dimJ = [1068 1102 177];
    centerJ = dimJ./2;
    verticesJ = [0 0 4; dimJ(1) 0 10; ...
        0 dimJ(2) 5; dimJ(1) dimJ(2) 15; ...
        0 0 104; dimJ(1) 0 110; ...
        0 dimJ(2) 105; dimJ(1) dimJ(2) 115];
    verticesJ = [verticesJ(:,1:2) verticesJ(:,3)-2]; % for fine tuning
elseif contains(name,'610')
    dimJ = [1024 1024 168];
    centerJ = dimJ./2;
    verticesJ = [0 0 10; dimJ(1) 0 39; ...
        0 dimJ(2) 0; dimJ(1) dimJ(2) 30; ...
        0 0 110; dimJ(1) 0 139; ...
        0 dimJ(2) 100; dimJ(1) dimJ(2) 130];
    verticesJ = [verticesJ(:,1:2) verticesJ(:,3)-2]; % for fine tuning
elseif contains(name,'620')
    dimJ = [1024 1024 168];
    centerJ = dimJ./2;
    verticesJ = [0 0 2; dimJ(1) 0 4; ...
        0 dimJ(2) 16; dimJ(1) dimJ(2) 19; ...
        0 0 102; dimJ(1) 0 104; ...
        0 dimJ(2) 116; dimJ(1) dimJ(2) 119];
    verticesJ = [verticesJ(:,1:2) verticesJ(:,3)-0]; % for fine tuning
elseif contains(name,'630')
    dimJ = [1065 1045 178];
    centerJ = dimJ./2;
    verticesJ = [0 0 13; dimJ(1) 0 25; ...
        0 dimJ(2) 4; dimJ(1) dimJ(2) 13; ...
        0 0 113; dimJ(1) 0 125; ...
        0 dimJ(2) 104; dimJ(1) dimJ(2) 113];
    verticesJ = [verticesJ(:,1:2) verticesJ(:,3)-5]; % for fine tuning
elseif contains(name,'640')
    dimJ = [1218 1239 180];
    centerJ = dimJ./2;
    verticesJ = [0 0 23; dimJ(1) 0 43; ...
        0 dimJ(2) 9; dimJ(1) dimJ(2) 23; ...
        0 0 123; dimJ(1) 0 143; ...
        0 dimJ(2) 109; dimJ(1) dimJ(2) 123];
    verticesJ = [verticesJ(:,1:2) verticesJ(:,3)-0]; % for fine tuning
elseif contains(name,'650')
    dimJ = [1052 1036 191];
    centerJ = dimJ./2;
    verticesJ = [0 0 0; dimJ(1) 0 0; ...
        0 dimJ(2) 18; dimJ(1) dimJ(2) 18; ...
        0 0 100; dimJ(1) 0 100; ...
        0 dimJ(2) 118; dimJ(1) dimJ(2) 118];
    verticesJ = [verticesJ(:,1:2) verticesJ(:,3)-0]; % for fine tuning
elseif contains(name,'66') %660 quads created manually
    dimJ = [520 520 185];
    centerJ = dimJ./2;
    verticesJ = [0 0 0; dimJ(1) 0 0; ...
        0 dimJ(2) 0; dimJ(1) dimJ(2) 0; ...
        0 0 100; dimJ(1) 0 100; ...
        0 dimJ(2) 100; dimJ(1) dimJ(2) 100];
    verticesJ = [verticesJ(:,1:2) verticesJ(:,3)-0]; % for fine tuning
elseif contains(name,'670')
    dimJ = [512 512 203];
    centerJ = dimJ./2;
    verticesJ = [0 0 10; dimJ(1) 0 13; ...
        0 dimJ(2) 10; dimJ(1) dimJ(2) 15; ...
        0 0 110; dimJ(1) 0 113; ...
        0 dimJ(2) 110; dimJ(1) dimJ(2) 115];
    verticesJ = [verticesJ(:,1:2) verticesJ(:,3)-0]; % for fine tuning
elseif contains(name,'680')
    dimJ = [1024 1024 185];
    centerJ = dimJ./2;
    verticesJ = [0 0 0; dimJ(1) 0 13; ...
        0 dimJ(2) 1; dimJ(1) dimJ(2) 14; ...
        0 0 100; dimJ(1) 0 113; ...
        0 dimJ(2) 101; dimJ(1) dimJ(2) 114];
    verticesJ = [verticesJ(:,1:2) verticesJ(:,3)-0]; % for fine tuning
elseif contains(name,'690')
    dimJ = [1064 1033 190];
    centerJ = dimJ./2;
    verticesJ = [0 0 9; dimJ(1) 0 21; ...
        0 dimJ(2) 0; dimJ(1) dimJ(2) 9; ...
        0 0 109; dimJ(1) 0 121; ...
        0 dimJ(2) 100; dimJ(1) dimJ(2) 109];
    verticesJ = [verticesJ(:,1:2) verticesJ(:,3)-0]; % for fine tuning
elseif contains(name,'700')
    dimJ = [1036 1044 193];
    centerJ = dimJ./2;
    verticesJ = [0 0 15; dimJ(1) 0 15; ...
        0 dimJ(2) 2; dimJ(1) dimJ(2) 2; ...
        0 0 115; dimJ(1) 0 115; ...
        0 dimJ(2) 102; dimJ(1) dimJ(2) 102];
    verticesJ = [verticesJ(:,1:2) verticesJ(:,3)-2]; % for fine tuning
elseif contains(name,'710')
    dimJ = [526 526 194];
    centerJ = dimJ./2;
    verticesJ = [0 0 32; dimJ(1) 0 41; ...
        0 dimJ(2) 21; dimJ(1) dimJ(2) 32; ...
        0 0 132; dimJ(1) 0 141; ...
        0 dimJ(2) 121; dimJ(1) dimJ(2) 132];
    verticesJ = [verticesJ(:,1:2) verticesJ(:,3)-0]; % for fine tuning
elseif contains(name,'720')
    dimJ = [526 526 194];
    centerJ = dimJ./2;
    verticesJ = [0 0 7; dimJ(1) 0 21; ...
        0 dimJ(2) 2; dimJ(1) dimJ(2) 7; ...
        0 0 107; dimJ(1) 0 121; ...
        0 dimJ(2) 102; dimJ(1) dimJ(2) 107];
    verticesJ = [verticesJ(:,1:2) verticesJ(:,3)-0]; % for fine tuning
elseif contains(name,'800')
    dimJ = [512 512 194];
    centerJ = dimJ./2;
    verticesJ = [0 0 13; dimJ(1) 0 14; ...
        0 dimJ(2) 14; dimJ(1) dimJ(2) 18; ...
        0 0 113; dimJ(1) 0 114; ...
        0 dimJ(2) 114; dimJ(1) dimJ(2) 118];
    verticesJ = [verticesJ(:,1:2) verticesJ(:,3)+8]; % for fine tuning
elseif contains(name,'810')
    dimJ = [512 512 194];
    centerJ = dimJ./2;
    verticesJ = [0 0 5; dimJ(1) 0 12; ...
        0 dimJ(2) 12; dimJ(1) dimJ(2) 20; ...
        0 0 105; dimJ(1) 0 112; ...
        0 dimJ(2) 112; dimJ(1) dimJ(2) 120];
    verticesJ = [verticesJ(:,1:2) verticesJ(:,3)+4]; % for fine tuning
elseif contains(name,'644')
    dimJ = [609 619.5 180];
    centerJ = dimJ./2;
    verticesJ = [0 0 3; dimJ(1) 0 8; ...
        0 dimJ(2) 0; dimJ(1) dimJ(2) 3; ...
        0 0 103; dimJ(1) 0 108; ...
        0 dimJ(2) 100; dimJ(1) dimJ(2) 103];
    verticesJ = [verticesJ(:,1:2) verticesJ(:,3)+23]; % for fine tuning
elseif contains(name,'643')
    dimJ = [609 619.5 180];
    centerJ = dimJ./2;
    verticesJ = [0 0 6; dimJ(1) 0 15; ...
        0 dimJ(2) 4; dimJ(1) dimJ(2) 6; ...
        0 0 106; dimJ(1) 0 115; ...
        0 dimJ(2) 104; dimJ(1) dimJ(2) 106];
    verticesJ = [verticesJ(:,1:2) verticesJ(:,3)+5]; % for fine tuning
elseif contains(name,'642')
    dimJ = [609 619.5 180];
    centerJ = dimJ./2;
    verticesJ = [0 0 12; dimJ(1) 0 25; ...
        0 dimJ(2) 5; dimJ(1) dimJ(2) 14; ...
        0 0 112; dimJ(1) 0 125; ...
        0 dimJ(2) 102; dimJ(1) dimJ(2) 114];
    verticesJ = [verticesJ(:,1:2) verticesJ(:,3)+25]; % for fine tuning
elseif contains(name,'830')
    dimJ = [1045 1052 170];
    centerJ = dimJ./2;
    verticesJ = [0 0 6; dimJ(1) 0 10; ...
        0 dimJ(2) 0; dimJ(1) dimJ(2) 6; ...
        0 0 106; dimJ(1) 0 110; ...
        0 dimJ(2) 100; dimJ(1) dimJ(2) 106];
    verticesJ = [verticesJ(:,1:2) verticesJ(:,3)-3]; % for fine tuning
elseif contains(name,'840')
    dimJ = [1046 1039 170];
    centerJ = dimJ./2;
    verticesJ = [0 0 10; dimJ(1) 0 10; ...
        0 dimJ(2) 0; dimJ(1) dimJ(2) 0; ...
        0 0 110; dimJ(1) 0 110; ...
        0 dimJ(2) 100; dimJ(1) dimJ(2) 100];
    verticesJ = [verticesJ(:,1:2) verticesJ(:,3)+0]; % for fine tuning
elseif contains(name,'820')
    dimJ = [1034 1036 189];
    centerJ = dimJ./2;
    verticesJ = [0 0 14; dimJ(1) 0 14; ...
        0 dimJ(2) 0; dimJ(1) dimJ(2) 0; ...
        0 0 114; dimJ(1) 0 114; ...
        0 dimJ(2) 100; dimJ(1) dimJ(2) 100];
    verticesJ = [verticesJ(:,1:2) verticesJ(:,3)+0]; % for fine tuning
    %% MATN4 counts
elseif contains(name,'1R1')
    dimJ = [1024 1024 175];
    centerJ = dimJ./2;
    verticesJ = [0 0 4; dimJ(1) 0 7; ...
        0 dimJ(2) 7; dimJ(1) dimJ(2) 10; ...
        0 0 104; dimJ(1) 0 107; ...
        0 dimJ(2) 107; dimJ(1) dimJ(2) 110];
    verticesJ = [verticesJ(:,1:2) verticesJ(:,3)-0]; % for fine tuning
elseif contains(name,'1R2')
    dimJ = [1024 1024 168];
    centerJ = dimJ./2;
    verticesJ = [0 0 10; dimJ(1) 0 10; ...
        0 dimJ(2) 7; dimJ(1) dimJ(2) 7; ...
        0 0 110; dimJ(1) 0 110; ...
        0 dimJ(2) 107; dimJ(1) dimJ(2) 107];
    verticesJ = [verticesJ(:,1:2) verticesJ(:,3)-0]; % for fine tuning
elseif contains(name,'3R1')
    dimJ = [1024 1024 168];
    centerJ = dimJ./2;
    verticesJ = [0 0 14; dimJ(1) 0 14; ...
        0 dimJ(2) 14; dimJ(1) dimJ(2) 14; ...
        0 0 114; dimJ(1) 0 114; ...
        0 dimJ(2) 114; dimJ(1) dimJ(2) 114];
    verticesJ = [verticesJ(:,1:2) verticesJ(:,3)-0]; % for fine tuning
elseif contains(name,'3R2')
    dimJ = [1024 1024 168];
    centerJ = dimJ./2;
    verticesJ = [0 0 23; dimJ(1) 0 23; ...
        0 dimJ(2) 15; dimJ(1) dimJ(2) 15; ...
        0 0 123; dimJ(1) 0 123; ...
        0 dimJ(2) 115; dimJ(1) dimJ(2) 115];
    verticesJ = [verticesJ(:,1:2) verticesJ(:,3)-0]; % for fine tuning
elseif contains(name,'3R3')
    dimJ = [1024 1024 168];
    centerJ = dimJ./2;
    verticesJ = [0 0 14; dimJ(1) 0 25; ...
        0 dimJ(2) 14; dimJ(1) dimJ(2) 25; ...
        0 0 114; dimJ(1) 0 125; ...
        0 dimJ(2) 114; dimJ(1) dimJ(2) 125];
    verticesJ = [verticesJ(:,1:2) verticesJ(:,3)-0]; % for fine tuning
end
end