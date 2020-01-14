#!/usr/bin/perl
#
# Copyright 2018   Suwon Shon
# Usage: make_voxceleb1_sv.pl /voxceleb1/ data/.

if (@ARGV != 2) {
  print STDERR "Usage: $0 <path-to-voxceleb1-dir> <path-to-output>\n";
  print STDERR "e.g. $0 /voxceleb1/ data\n";
  exit(1);
}
($db_base, $out_base_dir) = @ARGV;


# Generate voxceleb1 trials file for speaker verification task
$out_dir = "$out_base_dir/voxceleb1_trials";
$tmp_dir = "$out_dir";
if (system("mkdir -p $tmp_dir") != 0) {
  die "Error making directory $tmp_dir";
}

open(IN_TRIALS, "<", "local/voxceleb1_test.txt") or die "cannot open trials list";
open(OUT_TRIALS,">", "$out_dir/voxceleb1_trials_sv") or die "Could not open the output file $out_dir/voxceleb1_trials_sv";
while(<IN_TRIALS>) {
  chomp;
  ($is_target,$enrollment,$test) = split(",", $_);
  $target='nontarget';
  if ($is_target eq 1) {
    $target='target';
  }
  print OUT_TRIALS "$enrollment $test $target\n";

}
close(IN_TRIALS) || die;
close(OUT_TRIALS) || die;


# Generate voxceleb1 dev set
$out_dir = "$out_base_dir/voxceleb1_dev";
$tmp_dir = "$out_dir/tmp";
if (system("mkdir -p $tmp_dir") != 0) {
  die "Error making directory $tmp_dir";
}

open(IN_TRIALS, "<", "local/voxceleb1.csv") or die "cannot open trials list";
open(SPKR,">", "$tmp_dir/utt2spk") or die "Could not open the output file $out_dir/utt2spk";
open(WAV,">", "$tmp_dir/wav.scp") or die "Could not open the output file $out_dir/wav.scp";

while(<IN_TRIALS>) {
  chomp;
  ($filename,$utt,$start,$end,$spkr,$is_sv,$is_sid) = split(",", $_);
  if ($is_sv eq 'dev') {
    print WAV "$filename"," ${db_base}voxceleb1_wav/${filename}.wav\n";
    print SPKR "$filename $spkr\n";
  }
}

close(IN_TRIALS) || die;
# close(GNDR) || die;
close(SPKR) || die;
close(WAV) || die;

if (system(
  "cat $tmp_dir/utt2spk | sort | uniq >$out_dir/utt2spk") != 0) {
  die "Error creating utt2spk file in directory $out_dir";
}
if (system(
  "cat $tmp_dir/wav.scp | sort | uniq >$out_dir/wav.scp") != 0) {
  die "Error creating wav.scp file in directory $out_dir";
}

# Generate voxceleb1 test set
$out_dir = "$out_base_dir/voxceleb1_test";
$tmp_dir = "$out_dir/tmp";
if (system("mkdir -p $tmp_dir") != 0) {
  die "Error making directory $tmp_dir";
}
open(IN_TRIALS, "<", "local/voxceleb1_test.txt") or die "cannot open trials list";
open(SPKR,">", "$tmp_dir/utt2spk") or die "Could not open the output file $out_dir/utt2spk";
open(WAV,">", "$tmp_dir/wav.scp") or die "Could not open the output file $out_dir/wav.scp";

while(<IN_TRIALS>) {
  chomp;
  # ($filename,$utt,$start,$end,$spkr,$is_sv,$is_sid) = split(",", $_);
  ($trials,$filename1,$filename2) = split(" ", $_);
  ($filename1,$temp) = split(".wav",$filename1);
  ($filename2,$temp) = split(".wav",$filename2);
  ($spkr1,$temp) = split("/",$filename1);
  ($spkr2,$temp) = split("/",$filename2);
  print WAV "$filename1"," ${db_base}voxceleb1_wav/${filename1}.wav\n";
  print SPKR "$filename1 $spkr1\n";
  print WAV "$filename2"," ${db_base}voxceleb1_wav/${filename2}.wav\n";
  print SPKR "$filename2 $spkr2\n";

}
close(IN_TRIALS) || die;
close(SPKR) || die;
close(WAV) || die;


if (system(
  "cat $tmp_dir/utt2spk | sort | uniq >$out_dir/utt2spk") != 0) {
  die "Error creating utt2spk file in directory $out_dir";
}
if (system(
  "cat $tmp_dir/wav.scp | sort | uniq >$out_dir/wav.scp") != 0) {
  die "Error creating wav.scp file in directory $out_dir";
}
