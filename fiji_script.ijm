// Keep the stacked multipage tif in a folder and point Fiji to it. Then Fiji will read it and export
// the frames to the target folder.
// Make sure there is only one TIF in the source folder. If there are more TIFFs all of them will be processed

dir1 = getDirectory("Choose Source Directory");
dir2 = getDirectory("Choose Destination Directory");
list = getFileList(dir1);


setBatchMode(true);
for (i=0; i<list.length; i++)
{
    if (endsWith(list[i], "tif")) //check if its a tif file before processing it
    {
    	print("Processing file: "+list[i]); //print the file being processed
    	showProgress(i+1, list.length);
    	open(dir1+list[i]);
    	print(list[i]);
		run("Image Sequence... ", "format=JPG save=["+File.separator+dir2+"]");
    }

}
print("Done!")