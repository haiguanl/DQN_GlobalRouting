#!/usr/bin/perl
###############################################################################
#
# 
# ISPD 2008 Global Routing Contest Evaluation Script
#
# Maintained by Cliff Sze (csze@austin.ibm.com)
#
###############################################################################

# 
# Logs:
# 2009/01/20   Created
#              - More than 90% of codes are originally from Dr. Philip Chong
#              (pchong@cadence.com) from Cadence Berkeley Lab
#              - Some formatting codes are from Mehmet C. Yildiz from IBM Austin
#

$spec_verbose = 0;
$via_cost = 1;

use Getopt::Std;

###############################################################################
# Process all the switches
# -h help
# -H noheader
# -n net name for an eps file
# -v verbose level
# 
getopts('Hhn:v:');

$spec_verbose = 1 if ( $opt_v > 1 );

if($#ARGV != 1 || $opt_h ) {
    print STDERR "Usage: $0 [input design] [routed result]\n";
    print STDERR "Options: -n [net name]    Make net.ps file\n";
    print STDERR "         -v [level]       Verbosity level (0-2)\n";
    print STDERR "         -h               this (help) message\n";
    print STDERR "         -H               do not print header\n";
    print STDERR "Notes: Tot OF - Total overflow \n";
    print STDERR "       Max OF - Maximum overflow \n";
    print STDERR "       WL     - Total wirelength of the solution \n";
    exit 0;
}

$inname = $ARGV[0];
die "ERROR missing input file" unless(-f $inname);
$routename = $ARGV[1];
die "ERROR missing route file" unless(-f $routename);

$innameO = $inname;
$routenameO = $routename;

if($inname =~ /\.gz$/) {
    $inname = "gzip -dc $inname |";
}
elsif($inname =~ /\.bz2$/) {
    $inname = "bzip2 -dc $inname |";
}
if($routename =~ /\.gz$/) {
    $routename = "gzip -dc $routename |";
}
elsif($routename =~ /\.bz2$/) {
    $routename = "bzip2 -dc $routename |";
}

###############################################################################
# Process input file
#
open INFILE, "$inname";

$_ = <INFILE>;
die "ERROR" unless(/^\s*grid\s+(\d+)\s+(\d+)\s+(\d+)\s*$/);
$gridx = $1;
$gridy = $2;
$layers = $3;

$_ = <INFILE>;
die "ERROR" unless(/^\s*vertical capacity\s+(.*)\s*$/);
$_ = $1;
@vcap = split ' ', $_;
die "ERROR" unless($#vcap == $layers - 1);

$_ = <INFILE>;
die "ERROR" unless(/^\s*horizontal capacity\s+(.*)\s*$/);
$_ = $1;
@hcap = split ' ', $_;
die "ERROR" unless($#hcap == $layers - 1);

@gcapl = ();
@gcapr = ();
@gcapt = ();
@gcapb = ();
@rcapl = ();
@rcapr = ();
@rcapt = ();
@rcapb = ();

for $i (0 .. ($gridx - 1)) {
  push @gcapl, ();
  push @gcapr, ();
  push @gcapt, ();
  push @gcapb, ();
  push @rcapl, ();
  push @rcapr, ();
  push @rcapt, ();
  push @rcapb, ();
  for $j (0 .. ($gridy - 1)) {
    push @{$gcapl[$i]}, ();
    push @{$gcapr[$i]}, ();
    push @{$gcapt[$i]}, ();
    push @{$gcapb[$i]}, ();
    push @{$rcapl[$i]}, ();
    push @{$rcapr[$i]}, ();
    push @{$rcapt[$i]}, ();
    push @{$rcapb[$i]}, ();
    for $k (0 .. ($layers - 1)) {
      push @{$gcapl[$i][$j]}, ($i != 0) ? $hcap[$k] : 0;
      push @{$gcapr[$i][$j]}, ($i != $gridx - 1) ? $hcap[$k] : 0;
      push @{$gcapt[$i][$j]}, ($j != $gridy - 1) ? $vcap[$k] : 0;
      push @{$gcapb[$i][$j]}, ($j != 0) ? $vcap[$k] : 0;
      push @{$rcapl[$i][$j]}, ($i != 0) ? $hcap[$k] : 0;
      push @{$rcapr[$i][$j]}, ($i != $gridx - 1) ? $hcap[$k] : 0;
      push @{$rcapt[$i][$j]}, ($j != $gridy - 1) ? $vcap[$k] : 0;
      push @{$rcapb[$i][$j]}, ($j != 0) ? $vcap[$k] : 0;
    }
  }
}

$_ = <INFILE>;
die "ERROR" unless(/^\s*minimum width\s+(.*)\s*$/);
$_ = $1;
@minw = split ' ', $_;
die "ERROR" unless($#minw == $layers - 1);

################
# get mininum wire spacing
$_ = <INFILE>;
die "ERROR" unless(/^\s*minimum spacing\s+(.*)\s*$/);
$_ = $1;
@mins = split ' ', $_;
die "ERROR" unless($#mins == $layers - 1);

################
# get mininum via spacing, however, it is not used
$_ = <INFILE>;
die "ERROR" unless(/^\s*via spacing\s+(.*)\s*$/);
$_ = $1;
@minv = split ' ', $_;
die "ERROR" unless($#minv == $layers - 1);

################
# get origin x,y / grid x,y
$_ = <INFILE>;
die "ERROR" unless(/^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$/);
$llx = $1;
$lly = $2;
$xsize = $3;
$ysize = $4;

do {
    $_ = <INFILE>;
} while(/^\s*$/);
die "ERROR" unless(/^\s*num net\s+(\d+)\s*$/);
$numnet = $1;

sub xytogrid($$) {
    my $x = shift;
    my $y = shift;
    return (int(($x - $llx) / $xsize), int(($y - $lly) / $ysize));
}

################
# For each net
$netCount = 0;
for $i (0 .. ($numnet - 1)) {
  $netCount++;
  $_ = <INFILE>;
  die "ERROR" unless(/^\s*(\S+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$/);
  $n = $1;
  $id = $2;
  $npin = $3;
  $nminw = $4;
  $netExistInInputFile{$n}=1;
  push @nlist, $n;
  $netCountFromName{$n} = $netCount;
  $netmap{$n} = $i;
  push @nid, $id;
  push @netnumpin, $npin;
  push @netminw, $nminw;
  push @pinx, [];
  push @piny, [];
  push @pinl, [];
  ################
  # get pins of each net
  for $j (1 .. $npin) {
    $_ = <INFILE>;
    die "ERROR" unless(/^\s*(\d+)\s+(\d+)\s+(\d+)\s*$/);
    push @{$pinx[$i]}, $1;
    push @{$piny[$i]}, $2;
    push @{$pinl[$i]}, $3 - 1;
    die "ERROR: layer must be non-zero positive, net $n net_id $id" unless $3 > 0;
  }
}

while(<INFILE>) {
    next unless(/\S/);
    die "ERROR" unless(/^\s*(\d+)\s*$/);
    $numblock = $1;
    for $i (1 .. $numblock) {
	$_ = <INFILE>;
	@t = split ' ', $_;
	$x1 = $t[0];
	$y1 = $t[1];
	$l1 = $t[2] - 1;
  die "ERROR: layer must be non-zero positive" unless $t[2] > 0;
	$x2 = $t[3];
	$y2 = $t[4];
	$l2 = $t[5] - 1;
  die "ERROR: layer must be non-zero positive" unless $t[5] > 0;
	$c = $t[6];
	print "DEBUG: blocked $x1 $y1 $l1 $x2 $y2 $l2 $c\n" if ( $spec_verbose > 0 );
	if($x1 != $x2) {
	    die "ERROR" unless($y1 == $y2 and $l1 == $l2);
	    if($x2 < $x1) {
		$t = $x1;
		$x1 = $x2;
		$x2 = $t;
	    }
	    die "ERROR" unless($x1 == $x2 - 1);
	    $gcapr[$x1][$y1][$l1] = $c;
	    $gcapl[$x2][$y2][$l2] = $c;
	    $rcapr[$x1][$y1][$l1] = $c;
	    $rcapl[$x2][$y2][$l2] = $c;
	}
	elsif($y1 != $y2) {
	    die "ERROR" unless($x1 == $x2 and $l1 == $l2);
	    if($y2 < $y1) {
		$t = $y1;
		$y1 = $y2;
		$y2 = $t;
	    }
	    die "ERROR" unless($y1 == $y2 - 1);
	    $gcapt[$x1][$y1][$l1] = $c;
	    $gcapb[$x2][$y2][$l2] = $c;
	    $rcapt[$x1][$y1][$l1] = $c;
	    $rcapb[$x2][$y2][$l2] = $c;
	}
	else {
	    die"ERROR" ;
	}
    }
}

close INFILE;
#
###############################################################################

dump_all_rcap();

###############################################################################
# Process route file
# 
$ni = 0;
while($ni <= $#nlist) {
    push @netroutes, "";
    ++$ni;
}
open RFILE, "$routename";
while(<RFILE>) {
    next if /^\s*$/;
    @t = /^\s*(\S+)\s+(\d+)(?:\s+(\d+))?\s*$/;
    die "ERROR bad line" unless(@t);
    $n = $t[0];
    $id = $t[1];
    $rs = $t[2];
    die "ERROR net $n not found" if ( $netExistInInputFile{$n} eq 0 );
    $ni = $netCountFromName{$n} - 1;
    die "ERROR net $n not found" if ( $ni < 0 );
    print STDERR "WARNING net $n wrong id\n" if($nid[$ni] != $id and $opt_v);
    %cons = ();
    %endp = ();
    $routes = "";
    while(<RFILE>) {
      last if(/^\s*!\s*$/);
      --$rs if(defined $rs);
      die "ERROR net $n bad route" unless(/\s*\((\d+),(\d+),(\d+)\)-\((\d+),(\d+),(\d+)\)/);
      $x1 = $1;
      $y1 = $2;
      $l1 = $3 - 1;
      $x2 = $4;
      $y2 = $5;
      $l2 = $6 - 1;
      ($x1, $y1) = xytogrid($x1, $y1);
      ($x2, $y2) = xytogrid($x2, $y2);
      if($x1 != $x2) {
        die "ERROR net $n diagonal route" unless($y1 == $y2 and $l1 == $l2);
        if($x2 < $x1) {
          $t = $x1;
          $x1 = $x2;
          $x2 = $t;
        }
        $w = ($netminw[$ni] > $minw[$l1]) ? $netminw[$ni] : $minw[$l1];
        for $j ($x1 .. ($x2 - 1)) {
          $t = $j + 1;
          $s1 = "$j,$y1,$l1;";
          $s2 = "$t,$y1,$l1;";
          $cons{$s1} .= $s2;
          $cons{$s2} .= $s1;
          print "DEBUG ($j,$y1,$l1) $rcapr[$j][$y1][$l1] -= ($w + $mins[$l1])\n"  if ( $spec_verbose > 0 );
          $rcapr[$j][$y1][$l1] -= ($w + $mins[$l1]);
          print "DEBUG ($t,$y1,$l1) $rcapl[$t][$y1][$l1] -= ($w + $mins[$l1])\n"  if ( $spec_verbose > 0 );
          $rcapl[$t][$y1][$l1] -= ($w + $mins[$l1]);
        }
      }
      elsif($y1 != $y2) {
        die "ERROR net $n diagonal route" unless($x1 == $x2 and $l1 == $l2);
        if($y2 < $y1) {
          $t = $y1;
          $y1 = $y2;
          $y2 = $t;
        }
        $w = ($netminw[$ni] > $minw[$l1]) ? $netminw[$ni] : $minw[$l1];
        for $j ($y1 .. ($y2 - 1)) {
          $t = $j + 1;
          $s1 = "$x1,$j,$l1;";
          $s2 = "$x1,$t,$l1;";
          $cons{$s1} .= $s2;
          $cons{$s2} .= $s1;
          print "DEBUG ($x1,$j,$l1) $rcapt[$x1][$j][$l1] -= ($w + $mins[$l1])\n" if ( $spec_verbose > 0 );
          $rcapt[$x1][$j][$l1] -= ($w + $mins[$l1]);
          print "DEBUG ($x1,$t,$l1) $rcapb[$x1][$t][$l1] -= ($w + $mins[$l1])\n" if ( $spec_verbose > 0 );
          $rcapb[$x1][$t][$l1] -= ($w + $mins[$l1]);
        }
      }
      elsif($l1 != $l2) {
        die "ERROR net $n diagonal route" unless($x1 == $x2 and $y1 == $y2);
        if($l2 < $l1) {
          $t = $l1;
          $l1 = $l2;
          $l2 = $t;
        }
        for $j ($l1 .. ($l2 - 1)) {
          $t = $j + 1;
          $s1 = "$x1,$y1,$j;";
          $s2 = "$x1,$y1,$t;";
          $cons{$s1} .= $s2;
          $cons{$s2} .= $s1;
        }
      }
      else {
        die "ERROR net $n null route";
      }
      $routes .= "$x1,$y1,$l1,$x2,$y2,$l2;";
      $endp{"$x1,$y1,$l1;"} = 1;
      $endp{"$x2,$y2,$l2;"} = 1;
    }
    print STDERR "WARNING net $n bad route count\n" if($rs and $opt_v);
    if($netnumpin[$ni] <= 1000) {
      %visit = ();
      @vq = ();
      ($x1, $y1) = xytogrid($pinx[$ni][0], $piny[$ni][0]);
      $l1 = $pinl[$ni][0];
      $t = "$x1,$y1,$l1;";
      push @vq, $t;
      $visit{$t} = "START";
      $nottree = 0;
      %blind = ();
      while($t = pop @vq) {
          @cl = split ';', $cons{$t};
          for $j (@cl) {
            $j .= ';';
            if(!defined $visit{$j}) {
                push @vq, $j;
                $visit{$j} = $t;
            }
            elsif($j ne $visit{$t}) {
                $nottree = 1;
            }
          }
          $blind{$t} = 1 if($#cl <= 0);
          delete $endp{$t};
      }
      for $j (0 .. ($netnumpin[$ni] - 1)) {
          $x1 = $pinx[$ni][$j];
          $y1 = $piny[$ni][$j];
          $l1 = $pinl[$ni][$j];
          ($xg, $yg) = xytogrid($x1, $y1);
          $t = "$xg,$yg,$l1;";
          if(!defined $visit{$t}) {
            $tl = $l1 + 1;
            print "net $n pin ($x1,$y1,$tl) not attached\n"
          }
          delete $blind{$t};
      }
      die "ERROR net $n disjoint" if(keys %endp);
      print STDERR "WARNING net $n has cycle\n" if($nottree and $opt_v);
      print STDERR "WARNING net $n has blind route\n" if(keys %blind and $opt_v);
    }

    if($opt_n eq $n) {
      create_eps_file();
    }
    $netroutes[$ni]=$routes;
}
close RFILE;
#
###############################################################################
undef %netCountFromName;
undef %netExistInInputFile;

$totov = 0;
$totnetlen = 0;
$ni = 0;
$hasunroute = 0;
for $i (@netroutes) {
    $name = $nlist[$ni];
    if($i eq "" and $netnumpin[$ni] <= 1000) {
      $minx = 9e9;
      $miny = 9e9;
      $maxx = -9e9;
      $maxy = -9e9;
      for $i (0 .. ($netnumpin[$ni] - 1)) {
          $xc = $pinx[$ni][$i];
          $yc = $piny[$ni][$i];
          ($x1, $y1) = xytogrid($xc, $yc);
          $minx = $x1 if($x1 < $minx);
          $miny = $y1 if($y1 < $miny);
          $maxx = $x1 if($x1 > $maxx);
          $maxy = $y1 if($y1 > $maxy);
      }
      if($minx != $maxx || $miny != $maxy) {
          print STDERR "ERROR net $name unrouted\n";
          $hasunroute = 1;
      }
    }
    $netlen = 0;
    $ov = 0;
    @t = split ';', $i;
    for $j (@t) {
      @r = split ',', $j;
      $x1 = $r[0];
      $y1 = $r[1];
      $l1 = $r[2];
      $x2 = $r[3];
      $y2 = $r[4];
      $l2 = $r[5];
      if($x1 < $x2) {
          for $k ($x1 .. ($x2 - 1)) {
            if($rcapr[$k][$y1][$l1] < 0) {
                $ov = 1;
            }
          }
          $netlen += ($x2 - $x1);
      }
      elsif($y1 < $y2) {
          for $k ($y1 .. ($y2 - 1)) {
            if($rcapt[$x1][$k][$l1] < 0) {
                $ov = 1;
            }
          }
          $netlen += ($y2 - $y1);
      }
      elsif($l1 < $l2) {
          $netlen += $via_cost * ($l2 - $l1);
      }
      else {
          die "ERROR inconsistent data";
      }
    }
    ++$totov if($ov);
    print "INFO netlen $name $netlen\n" if($opt_v > 1);
    $totnetlen += $netlen;
    ++$ni;
}

$oedge = 0;
$otot = 0;
$omax = 0;
for $i (0 .. ($gridx - 1)) {
  for $j (0 .. ($gridy - 1)) {
    for $k (0 .. ($layers - 1)) {
      if($rcapr[$i][$j][$k] < 0) {
        ++$oedge;
        $otot -= $rcapr[$i][$j][$k];
        $omax = (-$rcapr[$i][$j][$k]) if($omax < -$rcapr[$i][$j][$k]);
      }
      if($rcapb[$i][$j][$k] < 0) {
        ++$oedge;
        $otot -= $rcapb[$i][$j][$k];
        $omax = (-$rcapb[$i][$j][$k]) if($omax < -$rcapb[$i][$j][$k]);
      }
    }
  }
}

###############################################################################
# DUMP results
my $files = 'report.txt';
my $OUTFILE;

$new_report = 1;
if ($new_report) {
    $format = "format STDOUT = \n";    
    if ( !$opt_H ) {
    $format .= "@<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< @>>>>>>>>>>>> @>>>>>>>>>>> @>>>>>>>>>>>>> \n";
    $format .="'File Names(In, Out)','"."Tot OF"."','"."Max OF"."','"."WL"."'\n";
    }
    $format .= "@<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< @>>>>>>>>>>>> @>>>>>>>>>>> @>>>>>>>>>>>>> \n";
    $format .= "'"."$innameO, $routenameO "."','".$otot."','".$omax."','".$totnetlen."'\n";
    $write  .= "$otot,$totnetlen\n";
    $format .= ".";
    local $^W = 0;
    eval $format;
    eval $write; 
    
    open $OUTFILE, '>>', $files;
    print { $OUTFILE } $write;
    close $OUTFILE;

    # open(my $fh, '>', $filename);
    # print $fh "Results\n";
    # close $fh;

    write(STDOUT);
} else {
    print "total net length = $totnetlen\n";
    print "overflowed nets = $totov\n";
    print "overflowed edges = $oedge\n";
    print "total overflow = $otot\n";
    print "max overflow = $omax\n";
}
dump_all_rcap();
die "ERROR has unrouted net" if($hasunroute);

###############################################################################
###############################################################################
# SUBROUTINE
###############################################################################
###############################################################################

###############################################################################
# DUMP ALL Routing Capacity
# (Warning: It uses global variables !!!)
sub dump_all_rcap {
  return if ( $spec_verbose < 1 );
  for $k (0 .. ($layers - 1)) {
    for $i (0 .. ($gridx - 1)) {
      for $j (0 .. ($gridy - 1)) {
        # print " ($i,$j,$k) l $rcapl[$i][$j][$k] r $rcapr[$i][$j][$k] t $rcapt[$i][$j][$k] b $rcapb[$i][$j][$k] \n";
        printf ("DEBUG (%d,%d,%d) l %+d r %+d t %+d b %+d \n",$i,$j,$k,$rcapl[$i][$j][$k],$rcapr[$i][$j][$k],$rcapt[$i][$j][$k],$rcapb[$i][$j][$k]);
      }
    }
  }
}

###############################################################################
# create an eps file for the net $n
# (Warning: It uses global variables !!!)
sub create_eps_file {
  $minx = 1e29;
  $miny = 1e29;
  $maxx = -1e29;
  $maxy = -1e29;
  $pstext = "";
  for $i (0 .. ($netnumpin[$ni] - 1)) {
    $xc = $pinx[$ni][$i];
    $yc = $piny[$ni][$i];
    ($x1, $y1) = xytogrid($xc, $yc);
    $minx = $x1 if($x1 < $minx);
    $miny = $y1 if($y1 < $miny);
    $maxx = $x1 if($x1 > $maxx);
    $maxy = $y1 if($y1 > $maxy);
    $x1 = 100 * $x1 - 10;
    $y1 = 100 * $y1 - 10;
    $x2 = $x1 + 20;
    $y2 = $y1 + 20;
    $pstext .= "n $x1 $y1 m $x2 $y1 l $x2 $y2 l $x1 $y2 l c f\n";
    $y1 -= 10 * (++$pincount{"$x1$y1"});
    $pstext .= "$x1 $y1 m (\\($xc,$yc\\)) w\n";
  }
  $pstext .= "1 0 0 setrgbcolor\n";
  @t = split ';', $routes;
  for $i (@t) {
    @r = split ',', $i;
    $x1 = $r[0];
    $y1 = $r[1];
    $l1 = $r[2];
    $x2 = $r[3];
    $y2 = $r[4];
    $l2 = $r[5];
    $minx = $x1 if($x1 < $minx);
    $miny = $y1 if($y1 < $miny);
    $maxx = $x1 if($x1 > $maxx);
    $maxy = $y1 if($y1 > $maxy);
    $minx = $x2 if($x2 < $minx);
    $miny = $y2 if($y2 < $miny);
    $maxx = $x2 if($x2 > $maxx);
    $maxy = $y2 if($y2 > $maxy);
    if($l1 < $l2) {
      $x1 = 100 * $x1 - 5;
      $y1 = 100 * $y1 - 5;
      $x2 = $x1 + 10;
      $y2 = $y1 + 10;
      $pstext .= "n $x1 $y1 m $x2 $y1 l $x2 $y2 l $x1 $y2 l c f\n";
    }
    else {
      $x1 = 100 * $x1;
      $y1 = 100 * $y1;
      $x2 = 100 * $x2;
      $y2 = 100 * $y2;
      $pstext .= "n $x1 $y1 m $x2 $y2 l c s\n";
    }
  }
  $x1 = 100 * ($minx - 1);
  $y1 = 100 * ($miny - 1);
  $x2 = 100 * ($maxx + 1);
  $y2 = 100 * ($maxy + 1);
  open NETFILE, ">$n.eps";
  print NETFILE <<EOF;
%!PS-Adobe-2.0 EPSF-2.0
\%\%BoundingBox: $x1 $y1 $x2 $y2
/! { bind def } bind def
/c { closepath } !
/f { fill } !
/l { lineto } !
/m { moveto } !
/n { newpath } !
/s { stroke } !
/w { show } !
save
/Helvetica findfont 10 scalefont setfont
1 setlinewidth 0 0 1 setrgbcolor
$pstext
restore showpage
\%\%EOF
EOF
  close NETFILE;
}
#
###############################################################################

