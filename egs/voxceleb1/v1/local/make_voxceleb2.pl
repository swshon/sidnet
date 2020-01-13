#!/usr/bin/perl
#
# Copyright 2018   Suwon Shon
# Usage: make_voxceleb2_sv.pl /voxceleb2/ data/.

if (@ARGV != 2) {
  print STDERR "Usage: $0 <path-to-voxceleb2-dir> <path-to-output>\n";
  print STDERR "e.g. $0 /voxceleb2/ data\n";
  exit(1);
}
($db_base, $out_base_dir) = @ARGV;


$out_dir = "$out_base_dir/voxceleb2_dev";

$tmp_dir = "$out_dir/tmp";
if (system("mkdir -p $tmp_dir") != 0) {
  die "Error making directory $tmp_dir"; 
}

open(IN_TRIALS, "<", "$out_base_dir/temp/dev_aac.txt") or die "cannot open trials list";
open(GNDR,">", "$out_dir/spk2gender") or die "Could not open the output file $out_dir/spk2gender";
open(SPKR,">", "$out_dir/utt2spk") or die "Could not open the output file $out_dir/utt2spk";
open(WAV,">", "$out_dir/wav.scp") or die "Could not open the output file $out_dir/wav.scp";

while(<IN_TRIALS>) {
  chomp;
  $filelist = $_;
  ($temp,$filename)=split("/aac/",$_);
  ($utt_id,$temp) = split(".m4a",$filename);
  ($spkr,$temp,$temp1) = split("/",$utt_id);
  $wav = "ffmpeg -v 8 -i $filelist -f wav -acodec pcm_s16le -|";
  print WAV "$utt_id $wav\n";
  print SPKR "$utt_id $spkr\n";
  print GNDR "$spkr m\n";
}

close(IN_TRIALS) || die;
close(GNDR) || die;
close(SPKR) || die;
close(WAV) || die;


if (system(
  "utils/utt2spk_to_spk2utt.pl $out_dir/utt2spk >$out_dir/spk2utt") != 0) {
  die "Error creating spk2utt file in directory $out_dir";
}
system("utils/fix_data_dir.sh $out_dir");
if (system("utils/validate_data_dir.sh --no-text --no-feats $out_dir") != 0) {
  die "Error validating directory $out_dir";
}



$out_dir = "$out_base_dir/voxceleb2_test";


$tmp_dir = "$out_dir/tmp";
if (system("mkdir -p $tmp_dir") != 0) {
  die "Error making directory $tmp_dir"; 
}


open(IN_TRIALS, "<", "$out_base_dir/temp/test_aac.txt") or die "cannot open trials list";
open(GNDR,">", "$out_dir/spk2gender") or die "Could not open the output file $out_dir/spk2gender";
open(SPKR,">", "$out_dir/utt2spk") or die "Could not open the output file $out_dir/utt2spk";
open(WAV,">", "$out_dir/wav.scp") or die "Could not open the output file $out_dir/wav.scp";

while(<IN_TRIALS>) {
  chomp;
  $filelist = $_;
  ($temp,$filename)=split("/aac/",$_);
  ($utt_id,$temp) = split(".m4a",$filename);
  ($spkr,$temp,$temp1) = split("/",$utt_id);
  $wav = "ffmpeg -v 8 -i $filelist -f wav -acodec pcm_s16le -|";

  print WAV "$utt_id $wav\n";
  print SPKR "$utt_id $spkr\n";
  print GNDR "$spkr m\n";
}

close(IN_TRIALS) || die;
close(GNDR) || die;
close(SPKR) || die;
close(WAV) || die;

if (system(
  "utils/utt2spk_to_spk2utt.pl $out_dir/utt2spk >$out_dir/spk2utt") != 0) {
  die "Error creating spk2utt file in directory $out_dir";
}
system("utils/fix_data_dir.sh $out_dir");
if (system("utils/validate_data_dir.sh --no-text --no-feats $out_dir") != 0) {
  die "Error validating directory $out_dir";
}


$out_dir = "$out_base_dir/voxceleb2_test_1utt";

$tmp_dir = "$out_dir/tmp";
if (system("mkdir -p $tmp_dir") != 0) {
  die "Error making directory $tmp_dir"; 
}

open(IN_TRIALS, "<", "$out_base_dir/temp/test_aac.txt") or die "cannot open trials list";
open(GNDR,">", "$out_dir/spk2gender") or die "Could not open the output file $out_dir/spk2gender";
open(SPKR,">", "$out_dir/utt2spk") or die "Could not open the output file $out_dir/utt2spk";
open(WAV,">", "$out_dir/wav.scp") or die "Could not open the output file $out_dir/wav.scp";

while(<IN_TRIALS>) {
  chomp;
  $filelist = $_;
  ($temp,$filename)=split("/aac/",$_);
  ($utt_id,$temp) = split(".m4a",$filename);
  ($spkr,$temp,$temp1) = split("/",$utt_id);
  $wav = "ffmpeg -v 8 -i $filelist -f wav -acodec pcm_s16le -|";
  print WAV "$utt_id $wav\n";
  print SPKR "$utt_id $utt_id\n";
  print GNDR "$utt_id m\n";
}

close(IN_TRIALS) || die;
close(GNDR) || die;
close(SPKR) || die;
close(WAV) || die;




if (system(
  "utils/utt2spk_to_spk2utt.pl $out_dir/utt2spk >$out_dir/spk2utt") != 0) {
  die "Error creating spk2utt file in directory $out_dir";
}
system("utils/fix_data_dir.sh $out_dir");
if (system("utils/validate_data_dir.sh --no-text --no-feats $out_dir") != 0) {
  die "Error validating directory $out_dir";
}
