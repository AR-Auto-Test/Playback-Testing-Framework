package c.e.b.p000if;

import android.content.Context;
import android.media.CamcorderProfile;
import android.media.MediaRecorder;
import android.util.Log;
import android.util.Size;
import android.view.Surface;
import c.b.a.a.a;
import com.google.ar.sceneform.SceneView;
import java.io.File;
import java.io.IOException;

/* compiled from: VideoRecorderSceneform.java */
/* renamed from: c.e.b.if.q  reason: invalid package */
/* loaded from: classes2.dex */
public class q {

    /* renamed from: a  reason: collision with root package name */
    public static final int[] f4902a = {1, 8, 6, 5, 4};

    /* renamed from: c  reason: collision with root package name */
    public MediaRecorder f4904c;

    /* renamed from: d  reason: collision with root package name */
    public Size f4905d;

    /* renamed from: e  reason: collision with root package name */
    public SceneView f4906e;

    /* renamed from: f  reason: collision with root package name */
    public int f4907f;

    /* renamed from: g  reason: collision with root package name */
    public File f4908g;

    /* renamed from: h  reason: collision with root package name */
    public String f4909h;
    public File i;
    public Surface l;
    public Context m;
    public int j = 10000000;
    public int k = 30;

    /* renamed from: b  reason: collision with root package name */
    public boolean f4903b = false;

    public q(Context context) {
        this.m = context;
    }

    public final void a() {
        if (this.f4908g == null) {
            File file = new File(this.m.getCacheDir() + "/UniteAR");
            this.f4908g = file;
            if (!file.exists()) {
                this.f4908g.mkdirs();
            }
        }
        String str = this.f4909h;
        if (str == null || str.isEmpty()) {
            this.f4909h = "TempVideo";
        }
        this.i = new File(this.f4908g, a.v(new StringBuilder(), this.f4909h, ".mp4"));
    }

    public boolean b() {
        if (this.f4903b) {
            this.f4903b = false;
            Surface surface = this.l;
            if (surface != null) {
                this.f4906e.stopMirroringToSurface(surface);
                this.l = null;
            }
            this.f4904c.stop();
            this.f4904c.reset();
        } else {
            if (this.f4904c == null) {
                this.f4904c = new MediaRecorder();
            }
            try {
                a();
                c();
                Surface surface2 = this.f4904c.getSurface();
                this.l = surface2;
                this.f4906e.startMirroringToSurface(surface2, 0, 0, this.f4905d.getWidth(), this.f4905d.getHeight());
                this.f4903b = true;
            } catch (IOException e2) {
                Log.e("VideoRecorder", "Exception setting up recorder", e2);
            }
        }
        return this.f4903b;
    }

    public final void c() {
        this.f4904c.setVideoSource(2);
        if (b.j.c.a.a(this.m, "android.permission.RECORD_AUDIO") == 0) {
            this.f4904c.setAudioSource(1);
        }
        this.f4904c.setOutputFormat(2);
        this.f4904c.setOutputFile(this.i.getAbsolutePath());
        this.f4904c.setVideoEncodingBitRate(this.j);
        this.f4904c.setVideoFrameRate(this.k);
        this.f4904c.setVideoSize(this.f4905d.getWidth(), this.f4905d.getHeight());
        this.f4904c.setVideoEncoder(this.f4907f);
        if (b.j.c.a.a(this.m, "android.permission.RECORD_AUDIO") == 0) {
            this.f4904c.setAudioEncoder(3);
        }
        this.f4904c.prepare();
        try {
            this.f4904c.start();
        } catch (IllegalStateException e2) {
            StringBuilder x = a.x("Exception starting capture: ");
            x.append(e2.getMessage());
            Log.e("VideoRecorder", x.toString(), e2);
        }
    }

    public void d(int i, int i2) {
        CamcorderProfile camcorderProfile = CamcorderProfile.hasProfile(i) ? CamcorderProfile.get(i) : null;
        if (camcorderProfile == null) {
            int[] iArr = f4902a;
            int length = iArr.length;
            int i3 = 0;
            while (true) {
                if (i3 >= length) {
                    break;
                }
                int i4 = iArr[i3];
                if (CamcorderProfile.hasProfile(i4)) {
                    camcorderProfile = CamcorderProfile.get(i4);
                    break;
                }
                i3++;
            }
        }
        if (i2 == 2) {
            this.f4905d = new Size(camcorderProfile.videoFrameWidth, camcorderProfile.videoFrameHeight);
        } else {
            this.f4905d = new Size(camcorderProfile.videoFrameHeight, camcorderProfile.videoFrameWidth);
        }
        this.f4907f = camcorderProfile.videoCodec;
        this.j = camcorderProfile.videoBitRate;
        this.k = camcorderProfile.videoFrameRate;
    }
}