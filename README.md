# LoKr-to-LoRa-Converter
LoKr to LoRa Converter. LoKr (LyCORIS) and LoHa are mathematically different from standard LoRAs. This tool will allow you to convert LoKr to Lora.

Apps like Draw Things attempt to perform a conversion in the background when trying to import LoKr files. However, this converter often crashes or fails, especially with FLUX-based LoKr files. As a result, a 4KB "dead" file (empty, containing only header information) is created in the Models folder. That 4KB file is unusable; it's junk.
To solve this problem, we need to convert that LoKr file to the standard LoRA format that Draw Things can understand. To do this, we'll perform a "mathematical operation" (using SVD - Singular Value Decomposition).
Solution: LoKr -> LoRA Converter Script
This Python script takes that stubborn LoKr file, mathematically breaks it down, and converts it into a standard LoRA file that is 100% compatible with apps like Draw Things.

How to Use It?
Again, with your favorite "drag-and-drop" method:
Open the terminal.
Type python3 (leave a space).
Drag the lokr_convert.py file to the terminal.
Leave a space.
Drag that problematic LoKr file (it should have a .safetensors extension) that you downloaded from Civitai to the terminal.
Press Enter.
What will happen?
The script will run, solve the complex "Kronecker" mathematical blocks inside the file, and convert them into standard "Up/Down" LoRA blocks. When the process is finished, a new file named _fixed_lora.safetensors will be created next to the original file.
3. Uploading to Draw Things
Delete that 4KB corrupted file in the Draw Things folder.
Upload the new _fixed_lora.safetensors file created by the script to Draw Things.
Now it will work perfectly.
Note: The "rank" (compression ratio) is calculated automatically during this process. The file size may be slightly larger, which is normal. The important thing is that Draw Things can now recognize it as "true LoRA".

Requirements: torch and safeensors
