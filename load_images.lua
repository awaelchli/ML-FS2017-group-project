
require 'image'
require 'torch'

load_images = {}

function load_images.load(dir, ext)

	-- 1. Load all files in directory

	-- We process all files in the given dir, and add their full path
	-- to a Lua table.

	-- Create empty table to store file names:
	local files = {}

	-- Go over all files in directory. We use an iterator, paths.files().
	for file in paths.files(dir) do
	   -- We only load files that match the extension
	   if file:find(ext .. '$') then
	      -- and insert the ones we care about in our table
	      table.insert(files, paths.concat(dir,file))
	   end
	end

	-- Check files
	if #files == 0 then
	   error('given directory doesnt contain any files of type: ' .. ext)
	end

	----------------------------------------------------------------------
	-- 2. Sort file names

	-- We sort files alphabetically, it's quite simple with table.sort()

	table.sort(files, function (a,b) return a < b end)

	print('Found files:')
	print(files)

	----------------------------------------------------------------------
	-- 3. Finally we load images

	-- Go over the file list:
	local images = {}
	for i,file in ipairs(files) do
	   -- load each image
	   table.insert(images, image.load(file))
	end

	print('Loaded images:')
	print(images)

	return images

end

return load_images