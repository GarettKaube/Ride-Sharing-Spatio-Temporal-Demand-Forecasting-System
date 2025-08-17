$total_rows = 150000000
$base_url = "https://data.cityofchicago.org/resource/m6dm-c72p.csv"
$app_token = "<CHICAGO DATA API APP TOKEN>" 
$headers = @{ "X-App-Token" = $app_token }

$output_folder = "$pwd\taxidata"

# Create folder if needed
if (-Not (Test-Path $output_folder)) {
    New-Item -ItemType Directory -Path $output_folder
}

$url = "$base_url`?`%24order=trip_start_timestamp%20DESC&%24limit=$total_rows"



$file = "$output_folder\rides.csv"
wget -O $file -Uri $url -Headers $headers -TimeoutSec 60