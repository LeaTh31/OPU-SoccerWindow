Summary: The Viewer for the RoboCup Soccer 2D Simulation
Name: soccerwindow2
Version: 5.1.1
Release: 0
License: GPL
Group: Applications/Engineering
Source: %{name}-%{version}.tar.gz
#Patch:
URL: http://rctools.sourceforge.jp/
Packager: Hidehisa Akiyama <akky@users.sourceforge.jp>
BuildRoot: %{_tmppath}/%{name}-root

#Requires: wxGTK >= 2.6.1
Requires: qt >= 3.3.0
Requires: boost >= 1.32
Requires: librcsc >= 1.3.0
#BuildRequires: wxGTK-devel >= 2.6.1
BuildRequires: qt-devel >= 3.3.0
BuildRequires: boost-devel >= 1.32
BuildRequires: librcsc-devel >= 1.3.0

%description
soccerwindow2 is a viewer program for the RoboCup Soccer 2D
Simulation. soccerwindow2 has a compatibility with the official
soccer monitor and logplayer. Moreover, soccerwindow2 can work as a
visual debugger and debug server to help us to develop a soccer agent.

%prep
rm -rf %{buildroot}

%setup

#%patch -p1

%build
%configure
make %{?_smp_mflags}

%install
rm -rf ${buildroot}

%makeinstall

%clean
rm -rf %{buildroot}

%post

%postun

%files
%defattr(-,root,root)
%doc AUTHORS COPYING INSTALL NEWS README
%{_bindir}/*
%{_datadir}/*

%changelog

* Thu Sep 28 2006 Hidehisa Akiyama <akky@users.sourceforge.jp>
-created initial version.
