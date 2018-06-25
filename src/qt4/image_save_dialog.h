// -*-c++-*-

/*!
  \file image_save_dialog.h
  \brief Field Canvas image save control dialog class Header File.
*/

/*
 *Copyright:

 Copyright (C) Hidehisa AKIYAMA

 This code is free software; you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation; either version 3, or (at your option)
 any later version.

 This code is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this code; see the file COPYING.  If not, write to
 the Free Software Foundation, 675 Mass Ave, Cambridge, MA 02139, USA.

 *EndCopyright:
 */

/////////////////////////////////////////////////////////////////////

#ifndef SOCCERWINDOW2_QT_IMAGE_SAVE_DIALOG_H
#define SOCCERWINDOW2_QT_IMAGE_SAVE_DIALOG_H

#include <QDialog>

class QComboBox;
class QLineEdit;
class QProgressDialog;
class QShowEvent;
class QSpinBox;

class FieldCanvas;
class MainData;
class MainWindow;

//! image save manager
class ImageSaveDialog
    : public QDialog {

    Q_OBJECT

private:

    MainWindow * M_main_window;
    FieldCanvas * M_field_canvas;

    MainData & M_main_data;

    QSpinBox * M_start_cycle;
    QSpinBox * M_end_cycle;

    QLineEdit * M_name_prefix;
    QComboBox * M_format_choice;

    QLineEdit * M_saved_dir;

    int M_current_index;

public:

    ImageSaveDialog( MainWindow * main_window,
                     FieldCanvas * field_canvas,
                     MainData & main_data );

    ImageSaveDialog( MainWindow * main_window,
                     FieldCanvas * field_canvas,
                     MainData & main_data,
                     int current_index );

    ~ImageSaveDialog();

private:

    void createControls();
    //void createControls(QString dir);

    QWidget * createCycleSelectControls();
    QWidget * createFileNameControls();
    QWidget * createDirSelectControls();
    //QWidget * createDirSelectControls(QString dir);
    QLayout * createExecuteControls();

    void saveImage( const int start_cycle,
                    const int end_cycle,
                    const QString & saved_dir,
                    const QString & name_prefix,
                    const QString & format_name );

    void saveImage( const int current_index,
                    const QString & saved_dir,
                    const QString & name_prefix,
                    const QString & format_name );

protected:

    void showEvent( QShowEvent * event );


private slots:

    void selectSavedDir();
    //void selectSavedDir(QString dir);

public slots:

    void selectAllCycle();
    void executeSave();

};

#endif
