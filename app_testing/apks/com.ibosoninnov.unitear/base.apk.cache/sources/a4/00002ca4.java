package com.ibosoninnov.unitear;

import android.app.Activity;
import android.content.ComponentName;
import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.Context;
import android.content.ContextWrapper;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.Rect;
import android.media.MediaPlayer;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.os.Parcelable;
import android.provider.MediaStore;
import android.text.format.DateFormat;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.RelativeLayout;
import android.widget.VideoView;
import androidx.core.content.FileProvider;
import b.b.c.h;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Date;
import java.util.Objects;

/* loaded from: classes2.dex */
public class CapturePreview extends h {
    public Button u;
    public VideoView w;
    public String r = null;
    public String s = null;
    public Bitmap t = null;
    public Uri v = null;
    public boolean x = false;

    /* loaded from: classes2.dex */
    public class a implements View.OnTouchListener {
        public a() {
        }

        @Override // android.view.View.OnTouchListener
        public boolean onTouch(View view, MotionEvent motionEvent) {
            if (motionEvent.getAction() == 0) {
                StringBuilder x = c.b.a.a.a.x("videoview clicked ");
                x.append(CapturePreview.this.r);
                Log.d("CapturePreview", x.toString());
                CapturePreview capturePreview = CapturePreview.this;
                String str = capturePreview.r;
                if (str != null) {
                    Objects.requireNonNull(capturePreview);
                    Uri b2 = FileProvider.b(capturePreview, "com.ibosoninnov.unitear.provider", new File(str));
                    Intent intent = new Intent("android.intent.action.VIEW");
                    intent.setDataAndType(b2, "video/mp4");
                    intent.addFlags(1);
                    capturePreview.startActivity(intent);
                }
                CapturePreview capturePreview2 = CapturePreview.this;
                Uri uri = capturePreview2.v;
                if (uri != null) {
                    Objects.requireNonNull(capturePreview2);
                    Intent intent2 = new Intent("android.intent.action.VIEW");
                    intent2.setDataAndType(uri, "video/mp4");
                    intent2.addFlags(1);
                    capturePreview2.startActivity(intent2);
                    return false;
                }
                return false;
            }
            return false;
        }
    }

    /* loaded from: classes2.dex */
    public class b implements MediaPlayer.OnPreparedListener {
        public b(CapturePreview capturePreview) {
        }

        @Override // android.media.MediaPlayer.OnPreparedListener
        public void onPrepared(MediaPlayer mediaPlayer) {
            mediaPlayer.setLooping(true);
        }
    }

    /* loaded from: classes2.dex */
    public class c implements View.OnClickListener {
        public c() {
        }

        @Override // android.view.View.OnClickListener
        public void onClick(View view) {
            StringBuilder x = c.b.a.a.a.x("imageview clicked ");
            x.append(CapturePreview.this.s);
            Log.d("CapturePreview", x.toString());
            CapturePreview capturePreview = CapturePreview.this;
            String str = capturePreview.s;
            Objects.requireNonNull(capturePreview);
            Uri b2 = FileProvider.b(capturePreview, "com.ibosoninnov.unitear.provider", new File(str));
            Intent intent = new Intent("android.intent.action.VIEW");
            intent.setDataAndType(b2, "image/*");
            intent.addFlags(1);
            capturePreview.startActivity(intent);
        }
    }

    /* loaded from: classes2.dex */
    public class d implements View.OnClickListener {
        public d() {
        }

        @Override // android.view.View.OnClickListener
        public void onClick(View view) {
            CapturePreview.this.finish();
        }
    }

    /* loaded from: classes2.dex */
    public class e implements View.OnClickListener {
        public e() {
        }

        @Override // android.view.View.OnClickListener
        public void onClick(View view) {
            Bitmap bitmap;
            CapturePreview capturePreview = CapturePreview.this;
            String str = capturePreview.r;
            if (str != null) {
                try {
                    capturePreview.v(str);
                } catch (IOException e2) {
                    e2.printStackTrace();
                }
            }
            CapturePreview capturePreview2 = CapturePreview.this;
            if (capturePreview2.s == null || (bitmap = capturePreview2.t) == null) {
                return;
            }
            Objects.requireNonNull(capturePreview2);
            try {
                Date date = new Date();
                DateFormat.format("yyyy-MM-dd_hh:mm:ss", date);
                ContentResolver contentResolver = capturePreview2.getContentResolver();
                ContentValues contentValues = new ContentValues();
                contentValues.put("_display_name", "IMG" + date + ".jpg");
                contentValues.put("mime_type", "image/jpeg");
                contentValues.put("relative_path", Environment.DIRECTORY_PICTURES + File.separator + "UniteAR");
                Uri insert = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues);
                Objects.requireNonNull(insert);
                OutputStream openOutputStream = contentResolver.openOutputStream(insert);
                bitmap.compress(Bitmap.CompressFormat.JPEG, 90, openOutputStream);
                Objects.requireNonNull(openOutputStream);
                openOutputStream.close();
                capturePreview2.u.setEnabled(false);
                capturePreview2.u.setText(capturePreview2.getResources().getString(R.string.saved));
            } catch (Exception e3) {
                e3.printStackTrace();
            }
        }
    }

    /* loaded from: classes2.dex */
    public class f implements View.OnClickListener {
        public f() {
        }

        @Override // android.view.View.OnClickListener
        public void onClick(View view) {
            Activity activity;
            ArrayList<? extends Parcelable> arrayList;
            Activity activity2;
            ArrayList<? extends Parcelable> arrayList2;
            CapturePreview capturePreview = CapturePreview.this;
            String str = capturePreview.r;
            if (str != null) {
                Objects.requireNonNull(capturePreview);
                Uri b2 = FileProvider.b(capturePreview, "com.ibosoninnov.unitear.provider", new File(str));
                Intent action = new Intent().setAction("android.intent.action.SEND");
                action.putExtra("androidx.core.app.EXTRA_CALLING_PACKAGE", capturePreview.getPackageName());
                action.putExtra("android.support.v4.app.EXTRA_CALLING_PACKAGE", capturePreview.getPackageName());
                action.addFlags(524288);
                Context context = capturePreview;
                while (true) {
                    if (!(context instanceof ContextWrapper)) {
                        activity2 = null;
                        break;
                    } else if (context instanceof Activity) {
                        activity2 = (Activity) context;
                        break;
                    } else {
                        context = ((ContextWrapper) context).getBaseContext();
                    }
                }
                if (activity2 != null) {
                    ComponentName componentName = activity2.getComponentName();
                    action.putExtra("androidx.core.app.EXTRA_CALLING_ACTIVITY", componentName);
                    action.putExtra("android.support.v4.app.EXTRA_CALLING_ACTIVITY", componentName);
                }
                if (b2 != null) {
                    arrayList2 = new ArrayList<>();
                    arrayList2.add(b2);
                } else {
                    arrayList2 = null;
                }
                action.setType("video/mp4");
                if (!(arrayList2 != null && arrayList2.size() > 1)) {
                    action.setAction("android.intent.action.SEND");
                    if (arrayList2 != null && !arrayList2.isEmpty()) {
                        action.putExtra("android.intent.extra.STREAM", arrayList2.get(0));
                        b.j.b.d.D(action, arrayList2);
                    } else {
                        action.removeExtra("android.intent.extra.STREAM");
                        action.setClipData(null);
                        action.setFlags(action.getFlags() & (-2));
                    }
                } else {
                    action.setAction("android.intent.action.SEND_MULTIPLE");
                    action.putParcelableArrayListExtra("android.intent.extra.STREAM", arrayList2);
                    b.j.b.d.D(action, arrayList2);
                }
                capturePreview.startActivity(Intent.createChooser(action, "Share Video"));
            }
            CapturePreview capturePreview2 = CapturePreview.this;
            Uri uri = capturePreview2.v;
            if (uri != null) {
                Objects.requireNonNull(capturePreview2);
                Intent action2 = new Intent().setAction("android.intent.action.SEND");
                action2.putExtra("androidx.core.app.EXTRA_CALLING_PACKAGE", capturePreview2.getPackageName());
                action2.putExtra("android.support.v4.app.EXTRA_CALLING_PACKAGE", capturePreview2.getPackageName());
                action2.addFlags(524288);
                Context context2 = capturePreview2;
                while (true) {
                    if (!(context2 instanceof ContextWrapper)) {
                        activity = null;
                        break;
                    } else if (context2 instanceof Activity) {
                        activity = (Activity) context2;
                        break;
                    } else {
                        context2 = ((ContextWrapper) context2).getBaseContext();
                    }
                }
                if (activity != null) {
                    ComponentName componentName2 = activity.getComponentName();
                    action2.putExtra("androidx.core.app.EXTRA_CALLING_ACTIVITY", componentName2);
                    action2.putExtra("android.support.v4.app.EXTRA_CALLING_ACTIVITY", componentName2);
                }
                if (uri != null) {
                    ArrayList<? extends Parcelable> arrayList3 = new ArrayList<>();
                    arrayList3.add(uri);
                    arrayList = arrayList3;
                } else {
                    arrayList = null;
                }
                action2.setType("video/mp4");
                if (!(arrayList != null && arrayList.size() > 1)) {
                    action2.setAction("android.intent.action.SEND");
                    if (arrayList != null && !arrayList.isEmpty()) {
                        action2.putExtra("android.intent.extra.STREAM", arrayList.get(0));
                        b.j.b.d.D(action2, arrayList);
                    } else {
                        action2.removeExtra("android.intent.extra.STREAM");
                        action2.setClipData(null);
                        action2.setFlags(action2.getFlags() & (-2));
                    }
                } else {
                    action2.setAction("android.intent.action.SEND_MULTIPLE");
                    action2.putParcelableArrayListExtra("android.intent.extra.STREAM", arrayList);
                    b.j.b.d.D(action2, arrayList);
                }
                capturePreview2.startActivity(Intent.createChooser(action2, "Share Video"));
            }
            CapturePreview capturePreview3 = CapturePreview.this;
            String str2 = capturePreview3.s;
            if (str2 != null) {
                Objects.requireNonNull(capturePreview3);
                Intent intent = new Intent("android.intent.action.SEND");
                intent.setFlags(1);
                intent.setType("image/*");
                intent.putExtra("android.intent.extra.STREAM", FileProvider.b(capturePreview3, "com.ibosoninnov.unitear.provider", new File(str2)));
                capturePreview3.startActivity(Intent.createChooser(intent, "Share Image"));
            }
        }
    }

    @Override // b.b.c.h, b.q.b.d, androidx.activity.ComponentActivity, b.j.b.e, android.app.Activity
    public void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        setContentView(R.layout.activity_capture_preview);
        Bundle extras = getIntent().getExtras();
        if (extras != null) {
            if (extras.containsKey("videoUrl")) {
                this.r = extras.getString("videoUrl");
            }
            if (extras.containsKey("imageUrl")) {
                this.s = extras.getString("imageUrl");
            }
        }
        Log.d("CapturePreview", "Started");
        LinearLayout linearLayout = (LinearLayout) findViewById(R.id.buttonLayout);
        RelativeLayout relativeLayout = (RelativeLayout) findViewById(R.id.saveBtnContainer);
        VideoView videoView = (VideoView) findViewById(R.id.videopreview);
        this.w = videoView;
        videoView.setOnTouchListener(new a());
        this.w.setOnPreparedListener(new b(this));
        if (this.r != null) {
            StringBuilder x = c.b.a.a.a.x("video loaded ");
            x.append(this.r);
            Log.d("CapturePreview", x.toString());
            this.w.setVideoPath(this.r);
            this.w.start();
        }
        ImageView imageView = (ImageView) findViewById(R.id.imagepreview);
        imageView.setOnClickListener(new c());
        if (this.s != null) {
            this.w.setVisibility(8);
            imageView.setVisibility(0);
            File file = new File(this.s);
            if (file.exists()) {
                Bitmap decodeFile = BitmapFactory.decodeFile(file.getAbsolutePath());
                this.t = decodeFile;
                Bitmap decodeResource = BitmapFactory.decodeResource(getResources(), 2131165506);
                Rect rect = new Rect(0, 0, decodeResource.getWidth(), decodeResource.getHeight());
                int width = decodeFile.getWidth();
                int height = decodeFile.getHeight();
                Bitmap createBitmap = Bitmap.createBitmap(width, height, decodeFile.getConfig());
                int i = width - ((int) (width / 2.5f));
                int i2 = width - (width / 15);
                int i3 = height / 20;
                Rect rect2 = new Rect(i, i3, i2, (int) (((decodeResource.getHeight() / decodeResource.getWidth()) * (i2 - i)) + i3));
                Canvas canvas = new Canvas(createBitmap);
                canvas.drawBitmap(decodeFile, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, (Paint) null);
                canvas.drawBitmap(decodeResource, rect, rect2, (Paint) null);
                this.t = createBitmap;
                imageView.setImageBitmap(createBitmap);
            }
        }
        ((ImageButton) findViewById(R.id.backbuttonpreview)).setOnClickListener(new d());
        Button button = (Button) findViewById(R.id.saveButton);
        this.u = button;
        button.setOnClickListener(new e());
        ((Button) findViewById(R.id.shareButton)).setOnClickListener(new f());
    }

    @Override // b.q.b.d, android.app.Activity
    public void onPause() {
        super.onPause();
        this.x = true;
    }

    @Override // b.q.b.d, android.app.Activity
    public void onResume() {
        super.onResume();
        VideoView videoView = this.w;
        if (videoView == null || !this.x) {
            return;
        }
        this.x = false;
        String str = this.r;
        if (str != null) {
            videoView.setVideoPath(str);
        }
        Uri uri = this.v;
        if (uri != null) {
            this.w.setVideoURI(uri);
        }
        this.w.start();
    }

    public Uri v(String str) {
        FileChannel fileChannel;
        FileChannel fileChannel2;
        Uri uri = null;
        try {
            Date date = new Date();
            DateFormat.format("yyyy-MM-dd_hh:mm:ss", date);
            ContentResolver contentResolver = getContentResolver();
            ContentValues contentValues = new ContentValues();
            contentValues.put("_display_name", "VID" + date + ".jpg");
            contentValues.put("mime_type", "video/mp4");
            contentValues.put("relative_path", Environment.DIRECTORY_MOVIES + File.separator + "UniteAR");
            Uri insert = contentResolver.insert(MediaStore.Video.Media.EXTERNAL_CONTENT_URI, contentValues);
            try {
                Objects.requireNonNull(insert);
                FileOutputStream fileOutputStream = (FileOutputStream) contentResolver.openOutputStream(insert);
                File file = new File(str);
                try {
                    fileChannel = new FileInputStream(file).getChannel();
                    try {
                        fileChannel2 = fileOutputStream.getChannel();
                    } catch (Throwable th) {
                        th = th;
                        fileChannel2 = null;
                    }
                    try {
                        fileChannel2.transferFrom(fileChannel, 0L, fileChannel.size());
                        fileChannel.close();
                        fileChannel2.close();
                        fileOutputStream.close();
                        file.delete();
                        this.r = null;
                        this.v = insert;
                        Log.d("CapturePreview", "saved to videoUri " + insert.toString());
                        this.u.setEnabled(false);
                        this.u.setText(getResources().getString(R.string.saved));
                        return insert;
                    } catch (Throwable th2) {
                        th = th2;
                        if (fileChannel != null) {
                            fileChannel.close();
                        }
                        if (fileChannel2 != null) {
                            fileChannel2.close();
                        }
                        if (fileOutputStream != null) {
                            fileOutputStream.close();
                        }
                        file.delete();
                        this.r = null;
                        this.v = insert;
                        Log.d("CapturePreview", "saved to videoUri " + insert.toString());
                        this.u.setEnabled(false);
                        this.u.setText(getResources().getString(R.string.saved));
                        throw th;
                    }
                } catch (Throwable th3) {
                    th = th3;
                    fileChannel = null;
                    fileChannel2 = null;
                }
            } catch (Exception e2) {
                e = e2;
                uri = insert;
                Log.e("CapturePreview", e.toString());
                return uri;
            }
        } catch (Exception e3) {
            e = e3;
        }
    }
}