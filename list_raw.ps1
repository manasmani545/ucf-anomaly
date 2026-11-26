Get-ChildItem -Recurse -File -Include *.mp4,*.avi,*.mov,*.mkv,*.csv,*.json,*.txt,*.yaml,*.yml |
Where-Object { $_.FullName -notmatch '\\\.venv\\' -and $_.FullName -notmatch '\\__pycache__\\' } |
Select-Object FullName, Length |
Format-Table -AutoSize |
Out-File -FilePath raw_files.txt -Encoding UTF8
