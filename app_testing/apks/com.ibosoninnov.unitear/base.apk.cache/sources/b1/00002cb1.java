package com.ibosoninnov.unitear;

import android.animation.ObjectAnimator;
import android.animation.PropertyValuesHolder;
import android.app.ActivityManager;
import android.app.Dialog;
import android.app.NotificationManager;
import android.content.ActivityNotFoundException;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.IntentSender;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Point;
import android.graphics.drawable.ColorDrawable;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.media.MediaPlayer;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.text.format.DateFormat;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.MenuItem;
import android.view.MotionEvent;
import android.view.PixelCopy;
import android.view.View;
import android.view.ViewGroup;
import android.view.Window;
import android.view.WindowManager;
import android.view.animation.Animation;
import android.view.animation.AnimationUtils;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.RelativeLayout;
import android.widget.TextView;
import android.widget.Toast;
import androidx.drawerlayout.widget.DrawerLayout;
import b.b.c.g;
import b.b.c.h;
import c.e.b.ac;
import c.e.b.b1;
import c.e.b.bc;
import c.e.b.bf;
import c.e.b.cc;
import c.e.b.ef.f;
import c.e.b.fc;
import c.e.b.hd;
import c.e.b.p000if.g;
import c.e.b.p000if.j;
import c.e.b.p000if.l;
import c.e.b.p000if.p;
import c.e.b.p000if.q;
import c.e.b.yb;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.google.android.material.navigation.NavigationView;
import com.google.android.material.snackbar.Snackbar;
import com.google.android.play.core.appupdate.AppUpdateInfo;
import com.google.android.play.core.appupdate.AppUpdateManager;
import com.google.android.play.core.appupdate.AppUpdateManagerFactory;
import com.google.android.play.core.install.InstallState;
import com.google.android.play.core.install.InstallStateUpdatedListener;
import com.google.android.play.core.tasks.OnSuccessListener;
import com.google.ar.core.ArCoreApk;
import com.google.ar.sceneform.FrameTime;
import com.google.ar.sceneform.HitTestResult;
import com.google.ar.sceneform.Node;
import com.google.ar.sceneform.Scene;
import com.google.ar.sceneform.SceneView;
import com.google.ar.sceneform.SimpleSceneView;
import com.google.ar.sceneform.math.Quaternion;
import com.google.ar.sceneform.math.Vector3;
import com.google.ar.sceneform.rendering.ExternalTexture;
import com.google.ar.sceneform.rendering.Material;
import com.google.ar.sceneform.rendering.Renderable;
import com.google.ar.sceneform.rendering.ShapeFactory;
import com.google.ar.sceneform.rendering.ViewRenderable;
import com.google.ar.sceneform.ux.SimpleFragment;
import com.google.ar.sceneform.ux.SimpleTransformableNode;
import com.google.gson.Gson;
import com.ibosoninnov.unitear.ARGalleryActivity;
import com.ibosoninnov.unitear.ImageTrackingActivity;
import com.ibosoninnov.unitear.LoginWebviewActivity;
import com.ibosoninnov.unitear.R;
import com.ibosoninnov.unitear.activities.Help2Activity;
import com.ibosoninnov.unitear.activities.HelpActivity;
import f.b0;
import f.d0;
import f.v;
import f.x;
import f.y;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Base64;
import java.util.Date;
import java.util.Iterator;
import java.util.Locale;
import java.util.Objects;
import java.util.Timer;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;
import java.util.function.Function;
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

/* loaded from: classes2.dex */
public class ImageTrackingActivity extends h implements yb.e, c.e.b.gf.a, c.e.b.gf.b, SensorEventListener, Scene.OnUpdateListener, Scene.OnPeekTouchListener {
    public static final /* synthetic */ int r = 0;
    public hd A;
    public MediaPlayer A0;
    public q B;
    public RelativeLayout B0;
    public c.e.b.p000if.d C;
    public RelativeLayout C0;
    public Context D;
    public RelativeLayout D0;
    public SensorManager E;
    public RelativeLayout E0;
    public Sensor F;
    public RelativeLayout F0;
    public RelativeLayout G0;
    public ImageButton H0;
    public Mat I;
    public ImageButton I0;
    public Mat J;
    public ImageButton J0;
    public String K;
    public ImageButton K0;
    public int L;
    public ImageButton L0;
    public int M;
    public ImageButton M0;
    public int N;
    public DrawerLayout N0;
    public ImageView O0;
    public float[] P;
    public ImageView P0;
    public TextView Q0;
    public TextView R0;
    public TextView S0;
    public TextView T0;
    public TextView U0;
    public TextView V0;
    public boolean W;
    public Button W0;
    public boolean X;
    public AppUpdateManager X0;
    public boolean Y;
    public boolean Z;
    public boolean a0;
    public boolean b0;
    public boolean c0;
    public boolean d0;
    public boolean e0;
    public boolean f0;
    public boolean g0;
    public boolean h0;
    public boolean i0;
    public boolean j0;
    public boolean k0;
    public boolean l0;
    public boolean m0;
    public boolean n0;
    public TextView r0;
    public yb s;
    public AutoFitTextureView t;
    public Timer t0;
    public ExternalTexture u;
    public CVLib u0;
    public SimpleFragment v;
    public f v0;
    public SimpleSceneView w;
    public l w0;
    public Node x;
    public bf x0;
    public Node y;
    public Handler y0;
    public p z;
    public Handler z0;
    public Quaternion G = null;
    public c.e.b.hf.c H = null;
    public int O = 0;
    public float Q = 1.66f;
    public float R = 1080.0f;
    public float S = 1920.0f;
    public long T = 4000;
    public long U = 0;
    public long V = 0;
    public final ArrayList<c.e.b.hf.a> o0 = new ArrayList<>();
    public final ArrayList<c.e.b.hf.d> p0 = new ArrayList<>();
    public final ArrayList<c.e.b.hf.a> q0 = new ArrayList<>();
    public Timer s0 = new Timer();
    public String Y0 = "";
    public final BaseLoaderCallback Z0 = new a(this);
    public InstallStateUpdatedListener a1 = new InstallStateUpdatedListener() { // from class: c.e.b.l1
        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
        @Override // com.google.android.play.core.listener.StateUpdatedListener
        public final void onStateUpdate(InstallState installState) {
            final ImageTrackingActivity imageTrackingActivity = ImageTrackingActivity.this;
            Objects.requireNonNull(imageTrackingActivity);
            if (installState.installStatus() == 11) {
                Snackbar make = Snackbar.make(imageTrackingActivity.findViewById(16908290), "Update almost finished!", -2);
                make.setAction("restart", new View.OnClickListener() { // from class: c.e.b.i1
                    @Override // android.view.View.OnClickListener
                    public final void onClick(View view) {
                        ImageTrackingActivity.this.X0.completeUpdate();
                    }
                });
                Context context = imageTrackingActivity.D;
                Object obj = b.j.c.a.f2074a;
                make.setActionTextColor(context.getColor(R.color.colorPrimary));
                make.show();
                c.e.b.p000if.d dVar = imageTrackingActivity.C;
                dVar.f4872b.putBoolean("force_update", false);
                dVar.f4872b.apply();
            }
        }
    };

    /* loaded from: classes2.dex */
    public class a extends BaseLoaderCallback {
        public a(Context context) {
            super(context);
        }

        @Override // org.opencv.android.BaseLoaderCallback, org.opencv.android.LoaderCallbackInterface
        public void onManagerConnected(int i) {
            if (i == 0) {
                ImageTrackingActivity.this.W = true;
            } else {
                super.onManagerConnected(i);
            }
        }
    }

    /* loaded from: classes2.dex */
    public class b implements g.a {
        public b() {
        }

        @Override // c.e.b.p000if.g.a
        public void a() {
        }

        @Override // c.e.b.p000if.g.a
        public void b(String str) {
            ImageTrackingActivity imageTrackingActivity = ImageTrackingActivity.this;
            int i = ImageTrackingActivity.r;
            imageTrackingActivity.N();
            ImageTrackingActivity.this.A();
            ImageTrackingActivity.this.D0.setVisibility(8);
            ImageTrackingActivity.v(ImageTrackingActivity.this, "scanning");
        }
    }

    /* loaded from: classes2.dex */
    public class c implements f.e {
        public c() {
        }

        @Override // f.e
        public void a(f.d dVar, b0 b0Var) {
            if (b0Var.B()) {
                d0 d0Var = b0Var.f5730h;
                Objects.requireNonNull(d0Var);
                final Bitmap decodeStream = BitmapFactory.decodeStream(d0Var.B());
                new Handler(Looper.getMainLooper()).post(new Runnable() { // from class: c.e.b.k0
                    @Override // java.lang.Runnable
                    public final void run() {
                        ImageTrackingActivity.c cVar = ImageTrackingActivity.c.this;
                        Bitmap bitmap = decodeStream;
                        Objects.requireNonNull(cVar);
                        Mat mat = new Mat();
                        Utils.bitmapToMat(bitmap, mat);
                        ImageTrackingActivity imageTrackingActivity = ImageTrackingActivity.this;
                        int i = ImageTrackingActivity.r;
                        Objects.requireNonNull(imageTrackingActivity);
                        float max = Math.max(mat.width(), mat.height());
                        if (max > 300.0f) {
                            float f2 = max / 300.0f;
                            Imgproc.resize(mat, mat, new Size(mat.width() / f2, mat.height() / f2));
                        }
                        Imgproc.cvtColor(mat, mat, 11);
                        ImageTrackingActivity imageTrackingActivity2 = ImageTrackingActivity.this;
                        if (imageTrackingActivity2.u0.patternDetectorSetImageToDetectJNI(mat.getNativeObjAddr())) {
                            imageTrackingActivity2.a0 = true;
                        } else {
                            imageTrackingActivity2.L(imageTrackingActivity2.getResources().getString(R.string.warning), 5000);
                        }
                        ImageTrackingActivity.this.Z = false;
                        Log.d("ImageTrackingActivity", "loadPattern response success");
                    }
                });
                return;
            }
            Log.e("ImageTrackingActivity", "loadPattern response unsucessfull");
            ImageTrackingActivity.this.Z = false;
        }

        @Override // f.e
        public void b(f.d dVar, IOException iOException) {
            Log.e("ImageTrackingActivity", "loadPattern " + iOException);
            ImageTrackingActivity.this.Z = false;
        }
    }

    /* loaded from: classes2.dex */
    public class d implements g.a {

        /* loaded from: classes2.dex */
        public class a implements cc.a {
            public a() {
            }

            @Override // c.e.b.cc.a
            public void a(String str) {
                Log.e("ImageTrackingActivity", str);
            }

            @Override // c.e.b.cc.a
            public void b(final String str) {
                ImageTrackingActivity.this.runOnUiThread(new Runnable() { // from class: c.e.b.w0
                    /* JADX WARN: Code restructure failed: missing block: B:28:0x00f9, code lost:
                        if (com.ibosoninnov.unitear.ImageTrackingActivity.this.C.f4871a.getBoolean("gallery_updated", false) != false) goto L26;
                     */
                    @Override // java.lang.Runnable
                    /*
                        Code decompiled incorrectly, please refer to instructions dump.
                    */
                    public final void run() {
                        String str2;
                        ImageTrackingActivity.d.a aVar = ImageTrackingActivity.d.a.this;
                        String str3 = str;
                        Objects.requireNonNull(aVar);
                        if (str3.isEmpty()) {
                            return;
                        }
                        ImageTrackingActivity.this.H = (c.e.b.hf.c) new Gson().fromJson(str3, new ic(aVar).getType());
                        ImageTrackingActivity imageTrackingActivity = ImageTrackingActivity.this;
                        String str4 = imageTrackingActivity.H.app_version;
                        String[] split = str4.split("\\.");
                        boolean z = false;
                        if (split.length >= 3) {
                            int parseInt = Integer.parseInt(split[0]);
                            int parseInt2 = Integer.parseInt(split[1]);
                            int parseInt3 = Integer.parseInt(split[2]);
                            try {
                                str2 = imageTrackingActivity.D.getPackageManager().getPackageInfo(imageTrackingActivity.D.getPackageName(), 0).versionName;
                            } catch (PackageManager.NameNotFoundException e2) {
                                e2.printStackTrace();
                                str2 = "";
                            }
                            String[] split2 = str2.split("\\.");
                            if (split2.length >= 3) {
                                boolean z2 = Integer.parseInt(split2[0]) <= parseInt && Integer.parseInt(split2[1]) <= parseInt2 && Integer.parseInt(split2[2]) < parseInt3;
                                Log.d("AppDetailsModel", "Version " + str2 + " >= " + str4 + " Expired = " + z2);
                                c.e.b.p000if.d dVar = imageTrackingActivity.C;
                                dVar.f4872b.putBoolean("force_update", z2);
                                dVar.f4872b.apply();
                            }
                        }
                        if (ImageTrackingActivity.this.C.f4871a.getString("ar_gallery", "").equals(ImageTrackingActivity.this.H.gallery_id)) {
                            ImageTrackingActivity imageTrackingActivity2 = ImageTrackingActivity.this;
                            if (imageTrackingActivity2.D(imageTrackingActivity2.D, "storage.json")) {
                            }
                        }
                        ImageTrackingActivity imageTrackingActivity3 = ImageTrackingActivity.this;
                        String str5 = ac.f4547a.f4549c;
                        if (imageTrackingActivity3.D(imageTrackingActivity3, "storage.json")) {
                            imageTrackingActivity3.K(str5);
                        } else {
                            try {
                                FileOutputStream openFileOutput = imageTrackingActivity3.openFileOutput("storage.json", 0);
                                openFileOutput.write("{}".getBytes());
                                openFileOutput.close();
                                z = true;
                            } catch (IOException unused) {
                            }
                            if (z) {
                                imageTrackingActivity3.K(str5);
                            }
                        }
                        ImageTrackingActivity imageTrackingActivity4 = ImageTrackingActivity.this;
                        c.e.b.hf.c cVar = imageTrackingActivity4.H;
                        c.e.b.p000if.d dVar2 = imageTrackingActivity4.C;
                        dVar2.f4872b.putString("arContentServer", " https://api.unitear.com/");
                        dVar2.f4872b.apply();
                        c.e.b.p000if.d dVar3 = imageTrackingActivity4.C;
                        dVar3.f4872b.putBoolean("useAuth", true);
                        dVar3.f4872b.apply();
                        c.e.b.p000if.d dVar4 = imageTrackingActivity4.C;
                        dVar4.f4872b.putString("imageRecoServer", cVar.image_reco_url);
                        dVar4.f4872b.apply();
                        c.e.b.p000if.d dVar5 = imageTrackingActivity4.C;
                        dVar5.f4872b.putString("token", cVar.token);
                        dVar5.f4872b.apply();
                        c.e.b.p000if.d dVar6 = imageTrackingActivity4.C;
                        dVar6.f4872b.putLong("last_updated", System.currentTimeMillis());
                        dVar6.f4872b.apply();
                        imageTrackingActivity4.E();
                    }
                });
            }
        }

        public d() {
        }

        @Override // c.e.b.p000if.g.a
        public void a() {
            String sb;
            y a2;
            if (ImageTrackingActivity.this.Y0 == null) {
                sb = "https://api.unitear.com/app/app-details2";
            } else {
                StringBuilder x = c.b.a.a.a.x("https://api.unitear.com/app/app-details2/");
                x.append(ImageTrackingActivity.this.Y0);
                sb = x.toString();
            }
            ImageTrackingActivity imageTrackingActivity = ImageTrackingActivity.this;
            imageTrackingActivity.Y0 = imageTrackingActivity.C.f4871a.getString("custom_user", "");
            if (ImageTrackingActivity.this.Y0.length() != 0) {
                ImageTrackingActivity.this.G0.setVisibility(0);
            }
            cc ccVar = new cc(new a());
            Objects.requireNonNull(ccVar);
            v vVar = cc.f4613a;
            if (vVar != null) {
                vVar.f6122d.a();
            }
            v.b bVar = new v.b();
            TimeUnit timeUnit = TimeUnit.SECONDS;
            bVar.a(10L, timeUnit);
            bVar.c(10L, timeUnit);
            bVar.b(15L, timeUnit);
            cc.f4613a = new v(bVar);
            if (ac.f4547a.f4552f) {
                y.a aVar = new y.a();
                aVar.d(sb);
                a2 = aVar.a();
            } else {
                y.a aVar2 = new y.a();
                aVar2.d(sb);
                a2 = aVar2.a();
            }
            ((x) cc.f4613a.a(a2)).b(new bc(ccVar));
        }

        @Override // c.e.b.p000if.g.a
        public void b(String str) {
            ImageTrackingActivity.this.D0.setVisibility(8);
            ImageTrackingActivity.v(ImageTrackingActivity.this, "initial");
        }
    }

    /* loaded from: classes2.dex */
    public class e implements cc.a {
        public e() {
        }

        @Override // c.e.b.cc.a
        public void a(String str) {
        }

        @Override // c.e.b.cc.a
        public void b(String str) {
            BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(new File(ImageTrackingActivity.this.D.getFilesDir(), "storage.json")));
            bufferedWriter.write(str);
            bufferedWriter.close();
            ImageTrackingActivity imageTrackingActivity = ImageTrackingActivity.this;
            c.e.b.p000if.d dVar = imageTrackingActivity.C;
            dVar.f4872b.putString("ar_gallery", imageTrackingActivity.H.gallery_id);
            dVar.f4872b.apply();
            c.e.b.p000if.d dVar2 = ImageTrackingActivity.this.C;
            dVar2.f4872b.putBoolean("gallery_updated", true);
            dVar2.f4872b.apply();
        }
    }

    public static void v(ImageTrackingActivity imageTrackingActivity, final String str) {
        if (imageTrackingActivity.f0) {
            return;
        }
        final Dialog dialog = new Dialog(imageTrackingActivity.D);
        dialog.setContentView(R.layout.no_internet);
        if (Objects.equals(str, "initial")) {
            dialog.setCanceledOnTouchOutside(false);
        }
        dialog.setOnCancelListener(new DialogInterface.OnCancelListener() { // from class: c.e.b.f2
            @Override // android.content.DialogInterface.OnCancelListener
            public final void onCancel(DialogInterface dialogInterface) {
                ImageTrackingActivity imageTrackingActivity2 = ImageTrackingActivity.this;
                String str2 = str;
                Dialog dialog2 = dialog;
                Objects.requireNonNull(imageTrackingActivity2);
                if (Objects.equals(str2, "initial")) {
                    imageTrackingActivity2.J();
                } else {
                    dialog2.dismiss();
                }
            }
        });
        ((Button) dialog.findViewById(R.id.retryBtn)).setOnClickListener(new View.OnClickListener() { // from class: c.e.b.y0
            @Override // android.view.View.OnClickListener
            public final void onClick(View view) {
                ImageTrackingActivity imageTrackingActivity2 = ImageTrackingActivity.this;
                String str2 = str;
                Dialog dialog2 = dialog;
                Objects.requireNonNull(imageTrackingActivity2);
                if (Objects.equals(str2, "initial")) {
                    imageTrackingActivity2.J();
                } else {
                    imageTrackingActivity2.M();
                }
                dialog2.dismiss();
            }
        });
        dialog.setOnShowListener(new DialogInterface.OnShowListener() { // from class: c.e.b.u1
            @Override // android.content.DialogInterface.OnShowListener
            public final void onShow(DialogInterface dialogInterface) {
                Dialog dialog2 = dialog;
                int i = ImageTrackingActivity.r;
                ObjectAnimator ofPropertyValuesHolder = ObjectAnimator.ofPropertyValuesHolder(dialog2.getWindow().getDecorView(), PropertyValuesHolder.ofFloat("scaleX", StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 1.0f), PropertyValuesHolder.ofFloat("scaleY", StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 1.0f));
                ofPropertyValuesHolder.setDuration(500L);
                ofPropertyValuesHolder.start();
            }
        });
        dialog.show();
    }

    public final void A() {
        Log.d("ImageTrackingActivity", "goBackToIdleState ");
        this.M = 0;
        this.M0.setVisibility(0);
        this.S0.setVisibility(0);
        this.J0.setVisibility(0);
        this.N0.setDrawerLockMode(0);
        this.I0.setVisibility(8);
        this.V0.setVisibility(8);
        this.K0.setVisibility(8);
        if (this.b0) {
            return;
        }
        this.B0.setVisibility(0);
    }

    public final void B() {
        StringBuilder x = c.b.a.a.a.x("handleBackButton currentState = ");
        x.append(this.M);
        Log.d("ImageTrackingActivity", x.toString());
        boolean z = true;
        if (this.l0) {
            Toast.makeText(this.D, "Recording OFF", 1).show();
            x();
            return;
        }
        int i = this.M;
        if (i == 0) {
            int intValue = Integer.valueOf(this.C.f4871a.getInt("launchCount", 0)).intValue();
            long currentTimeMillis = ((System.currentTimeMillis() - Long.valueOf(this.C.f4871a.getLong("installedDate", 0L)).longValue()) / 1000) / 3600;
            Log.d("ImageTrackingActivity", "Time since installed = " + currentTimeMillis + " hrs");
            if (this.C.f4871a.getBoolean("askRating", false) || intValue < 3 || currentTimeMillis <= 96) {
                z = false;
            } else {
                new g.a(this.D).setCustomTitle(getLayoutInflater().inflate(R.layout.rate_layout, (ViewGroup) null)).setTitle(getResources().getString(R.string.rate_and_review)).setMessage(getResources().getString(R.string.rate_and_review_des)).setCancelable(false).setPositiveButton(getResources().getString(17039370), new DialogInterface.OnClickListener() { // from class: c.e.b.z0
                    @Override // android.content.DialogInterface.OnClickListener
                    public final void onClick(DialogInterface dialogInterface, int i2) {
                        ImageTrackingActivity imageTrackingActivity = ImageTrackingActivity.this;
                        c.e.b.p000if.d dVar = imageTrackingActivity.C;
                        dVar.f4872b.putBoolean("askRating", true);
                        dVar.f4872b.apply();
                        String packageName = imageTrackingActivity.getPackageName();
                        try {
                            imageTrackingActivity.startActivity(new Intent("android.intent.action.VIEW", Uri.parse("market://details?id=" + packageName)));
                        } catch (ActivityNotFoundException unused) {
                            imageTrackingActivity.startActivity(new Intent("android.intent.action.VIEW", Uri.parse("https://play.google.com/store/apps/details?id=" + packageName)));
                        }
                        dialogInterface.dismiss();
                    }
                }).setNegativeButton(getResources().getString(17039360), new DialogInterface.OnClickListener() { // from class: c.e.b.h2
                    @Override // android.content.DialogInterface.OnClickListener
                    public final void onClick(DialogInterface dialogInterface, int i2) {
                        ImageTrackingActivity imageTrackingActivity = ImageTrackingActivity.this;
                        c.e.b.p000if.d dVar = imageTrackingActivity.C;
                        dVar.f4872b.putBoolean("askRating", true);
                        dVar.f4872b.apply();
                        imageTrackingActivity.finish();
                    }
                }).setIcon(2131165463).show();
            }
            if (z) {
                return;
            }
            finish();
        } else if (i == 1) {
            N();
            A();
            I();
        } else if (i == 2) {
            this.O0.setVisibility(0);
            this.O0.setTag("1");
            I();
            this.A.y();
            this.E0.setVisibility(8);
            this.B0.setVisibility(0);
            A();
        } else if (i != 3) {
        } else {
            this.O0.setVisibility(0);
            this.O0.setTag("1");
            A();
            I();
            this.B0.setVisibility(0);
            this.E0.setVisibility(8);
            this.g0 = false;
        }
    }

    public final void C(boolean z) {
        int i = z ? 8 : 0;
        if (z) {
            ImageButton imageButton = this.L0;
            Object obj = b.j.c.a.f2074a;
            imageButton.setImageDrawable(getDrawable(R.drawable.ic_video_recording));
            this.O0.setVisibility(8);
            this.w0 = new l(this, this.w);
        } else {
            ImageButton imageButton2 = this.L0;
            Object obj2 = b.j.c.a.f2074a;
            imageButton2.setImageDrawable(getDrawable(R.drawable.camerabutton));
            l lVar = this.w0;
            if (lVar != null) {
                lVar.a();
            }
        }
        if (this.M == 3) {
            this.K0.setVisibility(i);
        }
        StringBuilder x = c.b.a.a.a.x("CurrentState = ");
        x.append(this.M);
        Log.d("HideUIForRecording", x.toString());
        if (this.M == 3 && !z) {
            this.O0.setVisibility(0);
        }
        if (this.M != 2 || z) {
            return;
        }
        this.K0.setVisibility(0);
        this.O0.setVisibility(0);
    }

    public boolean D(Context context, String str) {
        return new File(context.getFilesDir().getAbsolutePath() + "/" + str).exists();
    }

    public final void E() {
        StringBuilder x = c.b.a.a.a.x("loadAppDetailsFromLocal");
        x.append(this.C.f4871a.getString("arContentServer", ""));
        Log.i("ImageTrackingActivity", x.toString());
        if (!this.C.f4871a.getString("arContentServer", "").isEmpty()) {
            this.k0 = true;
            ac.f4547a.f4549c = this.C.f4871a.getString("arContentServer", "");
            ac.f4547a.f4548b = this.C.f4871a.getString("imageRecoServer", "");
            ac.f4547a.f4550d = this.C.f4871a.getString("token", "");
            ac acVar = ac.f4547a;
            acVar.f4551e = 330;
            acVar.f4552f = this.C.f4871a.getBoolean("useAuth", false);
            this.D0.setVisibility(8);
            if (!this.C.f4871a.getBoolean("isScanPressed2", false)) {
                final Dialog dialog = new Dialog(this, R.style.DialogTheme);
                View inflate = LayoutInflater.from(this).inflate(R.layout.scan_target_dialog, (ViewGroup) null);
                ((Button) inflate.findViewById(R.id.uploadBtn)).setOnClickListener(new View.OnClickListener() { // from class: c.e.b.j1
                    @Override // android.view.View.OnClickListener
                    public final void onClick(View view) {
                        ImageTrackingActivity imageTrackingActivity = ImageTrackingActivity.this;
                        Dialog dialog2 = dialog;
                        Objects.requireNonNull(imageTrackingActivity);
                        dialog2.dismiss();
                        c.e.b.p000if.d dVar = imageTrackingActivity.C;
                        dVar.f4872b.putBoolean("isScanPressed2", true);
                        dVar.f4872b.apply();
                        imageTrackingActivity.startActivity(new Intent(imageTrackingActivity, LoginWebviewActivity.class));
                    }
                });
                ((Button) inflate.findViewById(R.id.scanTarget)).setOnClickListener(new View.OnClickListener() { // from class: c.e.b.m1
                    @Override // android.view.View.OnClickListener
                    public final void onClick(View view) {
                        ImageTrackingActivity imageTrackingActivity = ImageTrackingActivity.this;
                        Dialog dialog2 = dialog;
                        Objects.requireNonNull(imageTrackingActivity);
                        dialog2.dismiss();
                        c.e.b.p000if.d dVar = imageTrackingActivity.C;
                        dVar.f4872b.putBoolean("isScanPressed2", true);
                        dVar.f4872b.apply();
                        imageTrackingActivity.M();
                        imageTrackingActivity.y();
                    }
                });
                dialog.setOnCancelListener(new DialogInterface.OnCancelListener() { // from class: c.e.b.j2
                    @Override // android.content.DialogInterface.OnCancelListener
                    public final void onCancel(DialogInterface dialogInterface) {
                        ImageTrackingActivity imageTrackingActivity = ImageTrackingActivity.this;
                        Objects.requireNonNull(imageTrackingActivity);
                        dialogInterface.dismiss();
                        c.e.b.p000if.d dVar = imageTrackingActivity.C;
                        dVar.f4872b.putBoolean("isScanPressed2", true);
                        dVar.f4872b.apply();
                        imageTrackingActivity.y();
                    }
                });
                dialog.setCancelable(true);
                dialog.setContentView(inflate);
                Window window = dialog.getWindow();
                Objects.requireNonNull(window);
                window.setBackgroundDrawable(new ColorDrawable(0));
                dialog.show();
                dialog.setCanceledOnTouchOutside(true);
                return;
            }
            y();
            return;
        }
        J();
    }

    public final void F(String str) {
        if (this.Z) {
            return;
        }
        this.Z = true;
        y.a aVar = new y.a();
        aVar.d(str);
        ((x) new v().a(aVar.a())).b(new c());
    }

    public void G() {
        Log.d("ImageTrackingActivity", "onImageTargetFound");
        this.M = 2;
        runOnUiThread(new Runnable() { // from class: c.e.b.h0
            @Override // java.lang.Runnable
            public final void run() {
                ImageTrackingActivity imageTrackingActivity = ImageTrackingActivity.this;
                imageTrackingActivity.B0.setVisibility(8);
                imageTrackingActivity.L(imageTrackingActivity.getResources().getString(R.string.target_found), 2000);
                imageTrackingActivity.E0.setVisibility(0);
                imageTrackingActivity.J0.setVisibility(8);
                imageTrackingActivity.V0.setVisibility(0);
                imageTrackingActivity.I0.setVisibility(0);
                imageTrackingActivity.M0.setVisibility(8);
                imageTrackingActivity.S0.setVisibility(8);
                imageTrackingActivity.O0.setVisibility(0);
                if (!imageTrackingActivity.l0) {
                    imageTrackingActivity.K0.setVisibility(0);
                }
                imageTrackingActivity.O0.setTag("2");
                b.v.u.c.z(imageTrackingActivity.D);
            }
        });
    }

    public void H(String str, boolean z) {
        hd hdVar = this.A;
        Iterator<Node> it = hdVar.n.iterator();
        while (it.hasNext()) {
            Node next = it.next();
            if (next.getName().equals("playPauseButton")) {
                try {
                    ((ImageView) ((ViewRenderable) next.getRenderable()).getView().findViewById(R.id.img_loader_view)).setImageResource(R.drawable.play);
                } catch (Exception e2) {
                    Log.e("LoaderARContent", e2.toString());
                }
            }
        }
        hdVar.r();
        Intent intent = new Intent(this.D, CapturePreview.class);
        if (z) {
            intent.putExtra("videoUrl", str);
        } else {
            intent.putExtra("imageUrl", str);
        }
        startActivity(intent);
    }

    public final void I() {
        Log.d("ImageTrackingActivity", "reload scene");
        this.P0.setVisibility(0);
        this.U0.setVisibility(0);
        this.a0 = false;
        this.d0 = false;
        this.g0 = false;
        this.e0 = false;
        this.X = false;
        this.m0 = false;
        this.u0.patternDetectorInitJNI();
        this.c0 = false;
        hd hdVar = this.A;
        if (hdVar != null) {
            hdVar.h();
        }
        this.y.setParent(null);
        w(this.w);
    }

    public final void J() {
        new c.e.b.p000if.g(this, new d()).execute(new Void[0]);
    }

    public void K(String str) {
        y a2;
        String q = c.b.a.a.a.q(str, "unitear/ground_plane_new");
        cc ccVar = new cc(new e());
        Objects.requireNonNull(ccVar);
        v vVar = cc.f4613a;
        if (vVar != null) {
            vVar.f6122d.a();
        }
        v.b bVar = new v.b();
        TimeUnit timeUnit = TimeUnit.SECONDS;
        bVar.a(10L, timeUnit);
        bVar.c(10L, timeUnit);
        bVar.b(15L, timeUnit);
        cc.f4613a = new v(bVar);
        if (ac.f4547a.f4552f) {
            y.a aVar = new y.a();
            aVar.d(q);
            a2 = aVar.a();
        } else {
            y.a aVar2 = new y.a();
            aVar2.d(q);
            a2 = aVar2.a();
        }
        ((x) cc.f4613a.a(a2)).b(new bc(ccVar));
    }

    public final void L(String str, int i) {
        Log.d("ImageTrackingActivity", "showToast - " + str);
        final TextView textView = (TextView) findViewById(R.id.msgTxt);
        textView.setBackground(null);
        textView.setText(str);
        Animation loadAnimation = AnimationUtils.loadAnimation(this.D, R.anim.translate_down);
        final Animation loadAnimation2 = AnimationUtils.loadAnimation(this.D, R.anim.translate_up);
        textView.clearAnimation();
        this.C0.setVisibility(0);
        textView.startAnimation(loadAnimation);
        this.z0.removeCallbacksAndMessages(null);
        this.z0.postDelayed(new Runnable() { // from class: c.e.b.i0
            @Override // java.lang.Runnable
            public final void run() {
                final ImageTrackingActivity imageTrackingActivity = ImageTrackingActivity.this;
                final TextView textView2 = textView;
                Animation animation = loadAnimation2;
                Objects.requireNonNull(imageTrackingActivity);
                textView2.startAnimation(animation);
                new Handler().postDelayed(new Runnable() { // from class: c.e.b.a1
                    @Override // java.lang.Runnable
                    public final void run() {
                        ImageTrackingActivity imageTrackingActivity2 = ImageTrackingActivity.this;
                        TextView textView3 = textView2;
                        imageTrackingActivity2.C0.setVisibility(8);
                        textView3.clearAnimation();
                    }
                }, 500L);
            }
        }, i);
    }

    public final void M() {
        this.F0.setVisibility(0);
        this.M = 1;
        this.O0.setTag("1");
        Log.d("ImageTrackingActivity", "startScan");
        this.d0 = false;
        this.j0 = true;
        this.R0.setVisibility(0);
        this.s0 = new Timer();
        this.M0.setVisibility(8);
        this.S0.setVisibility(8);
        this.y0.postDelayed(new Runnable() { // from class: c.e.b.p2
            @Override // java.lang.Runnable
            public final void run() {
                final ImageTrackingActivity imageTrackingActivity = ImageTrackingActivity.this;
                imageTrackingActivity.d0 = true;
                imageTrackingActivity.N();
                imageTrackingActivity.runOnUiThread(new Runnable() { // from class: c.e.b.n1
                    @Override // java.lang.Runnable
                    public final void run() {
                        ImageTrackingActivity imageTrackingActivity2 = ImageTrackingActivity.this;
                        imageTrackingActivity2.b0 = true;
                        imageTrackingActivity2.U0.setVisibility(8);
                        imageTrackingActivity2.B0.setVisibility(8);
                        imageTrackingActivity2.P0.setVisibility(8);
                    }
                });
                if (imageTrackingActivity.f0) {
                    return;
                }
                final Dialog dialog = new Dialog(imageTrackingActivity.D);
                Window window = dialog.getWindow();
                Objects.requireNonNull(window);
                window.setBackgroundDrawable(new ColorDrawable(0));
                dialog.setContentView(R.layout.no_targetimage_found);
                ((TextView) dialog.findViewById(R.id.unitear_btn)).setOnClickListener(new View.OnClickListener() { // from class: c.e.b.j0
                    @Override // android.view.View.OnClickListener
                    public final void onClick(View view) {
                        ImageTrackingActivity imageTrackingActivity2 = ImageTrackingActivity.this;
                        Dialog dialog2 = dialog;
                        Objects.requireNonNull(imageTrackingActivity2);
                        Intent intent = new Intent("android.intent.action.VIEW");
                        intent.setData(Uri.parse("https://www.unitear.com"));
                        imageTrackingActivity2.startActivity(intent);
                        dialog2.dismiss();
                    }
                });
                ((Button) dialog.findViewById(R.id.retryscan)).setOnClickListener(new View.OnClickListener() { // from class: c.e.b.s1
                    @Override // android.view.View.OnClickListener
                    public final void onClick(View view) {
                        ImageTrackingActivity imageTrackingActivity2 = ImageTrackingActivity.this;
                        Dialog dialog2 = dialog;
                        imageTrackingActivity2.M();
                        dialog2.dismiss();
                    }
                });
                dialog.show();
                imageTrackingActivity.N();
                dialog.setOnDismissListener(new DialogInterface.OnDismissListener() { // from class: c.e.b.q1
                    @Override // android.content.DialogInterface.OnDismissListener
                    public final void onDismiss(DialogInterface dialogInterface) {
                        ImageTrackingActivity imageTrackingActivity2 = ImageTrackingActivity.this;
                        imageTrackingActivity2.b0 = false;
                        imageTrackingActivity2.U0.setVisibility(0);
                        imageTrackingActivity2.B0.setVisibility(0);
                        imageTrackingActivity2.A();
                    }
                });
            }
        }, 28000L);
        new c.e.b.p000if.g(this, new b()).execute(new Void[0]);
    }

    public final void N() {
        Log.d("ImageTrackingActivity", "stopScan");
        this.j0 = false;
        this.y0.removeCallbacksAndMessages(null);
        runOnUiThread(new Runnable() { // from class: c.e.b.y1
            @Override // java.lang.Runnable
            public final void run() {
                ImageTrackingActivity imageTrackingActivity = ImageTrackingActivity.this;
                imageTrackingActivity.R0.setVisibility(8);
                imageTrackingActivity.F0.setVisibility(8);
            }
        });
    }

    public void O(String str) {
        String string = this.C.f4871a.getString("history", "");
        if (string != null && !string.isEmpty()) {
            str = c.b.a.a.a.r(string, ",", str);
        }
        Log.d("storeHistory jsonData", "jsonData:  " + str);
        c.e.b.p000if.d dVar = this.C;
        dVar.f4872b.putString("history", str);
        dVar.f4872b.apply();
    }

    public final void P() {
        Intent intent;
        if (this.h0 || this.i0) {
            this.g0 = true;
            if (this.A != null) {
                this.y.setEnabled(false);
                if (this.i0) {
                    intent = new Intent(this, NonARCoreActivitySceneform.class);
                } else {
                    intent = new Intent(this, ARCoreSceneformActivity.class);
                }
                hd hdVar = this.A;
                if (hdVar.B == null) {
                    L("Loading data.. Please wait", 2000);
                    return;
                }
                intent.putExtra("alphaid", hdVar.y);
                intent.putExtra("response", this.A.B);
                intent.addFlags(536870912);
                startActivity(intent);
                if (this.m0) {
                    return;
                }
                new Handler().postDelayed(new Runnable() { // from class: c.e.b.i2
                    @Override // java.lang.Runnable
                    public final void run() {
                        ImageTrackingActivity imageTrackingActivity = ImageTrackingActivity.this;
                        imageTrackingActivity.P0.setVisibility(0);
                        imageTrackingActivity.U0.setVisibility(0);
                    }
                }, 2000L);
            }
        }
    }

    public final void Q(String str, String str2) {
        Intent intent;
        if (this.h0 || this.i0) {
            if (this.i0) {
                intent = new Intent(this, NonARCoreActivitySceneform.class);
            } else {
                intent = new Intent(this, ARCoreSceneformActivity.class);
            }
            intent.putExtra("menuItemJson", str);
            intent.putExtra("id", str2);
            intent.putExtra("fromargallery", true);
            intent.addFlags(536870912);
            startActivity(intent);
        }
    }

    @Override // c.e.b.gf.a
    public void c(c.e.b.hf.a aVar) {
        this.A = new hd("", ac.f4547a.f4549c, this.y, this.v, this.D, this);
    }

    @Override // c.e.b.gf.a
    public void d(c.e.b.hf.a aVar) {
        File file = new File(getCacheDir(), "assets/models");
        if (!file.exists() && file.mkdir()) {
            L("Failed to Create Directory", 0);
        }
        if (new File(this.D.getCacheDir(), c.b.a.a.a.v(c.b.a.a.a.x("assets/models/"), aVar.id, ".glb")).exists()) {
            f(aVar.id, 101, false, "");
        } else {
            new j(this, aVar.id, false).execute(aVar.file_loc);
        }
    }

    @Override // c.e.b.gf.b
    public void f(String str, int i, boolean z, String str2) {
        if (z) {
            try {
                int size = this.q0.size();
                for (int i2 = 0; i2 <= size; i2++) {
                    if (this.q0.get(i2).id.equals(str)) {
                        this.q0.get(i2).downloadStatus = i;
                        this.v0.notifyItemChanged(i2);
                        return;
                    }
                }
            } catch (Exception e2) {
                e2.printStackTrace();
            }
        }
    }

    @Override // c.e.b.gf.a
    public void h(String str) {
        for (int i = 0; i < this.p0.size(); i++) {
            this.p0.get(i).isSelected = this.p0.get(i).name.equals(str);
        }
        this.q0.clear();
        Iterator<c.e.b.hf.a> it = this.o0.iterator();
        while (it.hasNext()) {
            c.e.b.hf.a next = it.next();
            if (next.category.equals(str)) {
                this.q0.add(next);
            }
            if (new File(getCacheDir(), c.b.a.a.a.v(c.b.a.a.a.x("models/"), next.id, ".glb")).exists()) {
                next.downloadStatus = 101;
            }
        }
        this.v0 = new f(this.q0, this);
    }

    @Override // android.hardware.SensorEventListener
    public void onAccuracyChanged(Sensor sensor, int i) {
    }

    @Override // b.q.b.d, android.app.Activity
    public void onActivityResult(int i, int i2, Intent intent) {
        super.onActivityResult(i, i2, intent);
        if (this.C.f4871a.getBoolean("force_update", false) && i == 1 && i2 != -1) {
            y();
        }
    }

    @Override // androidx.activity.ComponentActivity, android.app.Activity
    public void onBackPressed() {
        StringBuilder x = c.b.a.a.a.x("BackPressed ");
        x.append(this.M);
        Log.d("ImageTrackingActivity", x.toString());
        B();
        Log.d("ImageTrackingActivity", "BackPressed Finished " + this.M);
    }

    @Override // b.b.c.h, b.q.b.d, androidx.activity.ComponentActivity, b.j.b.e, android.app.Activity
    public void onCreate(Bundle bundle) {
        Intent intent;
        super.onCreate(bundle);
        q().r(1);
        getWindow().setFlags(1024, 1024);
        getWindow().addFlags(128);
        getWindow().addFlags(1536);
        setContentView(R.layout.activity_image_tracking);
        this.X0 = AppUpdateManagerFactory.create(this);
        this.C = new c.e.b.p000if.d(this);
        this.D = this;
        DisplayMetrics displayMetrics = new DisplayMetrics();
        getWindowManager().getDefaultDisplay().getMetrics(displayMetrics);
        float f2 = displayMetrics.heightPixels;
        this.S = f2;
        float f3 = displayMetrics.widthPixels;
        this.R = f3;
        this.Q = f2 / f3;
        this.u0 = new CVLib();
        this.F0 = (RelativeLayout) findViewById(R.id.scanlayout);
        this.K0 = (ImageButton) findViewById(R.id.refreshButton);
        this.I0 = (ImageButton) findViewById(R.id.groundplaneButton);
        this.V0 = (TextView) findViewById(R.id.groundplaneButtonTxt);
        this.P0 = (ImageView) findViewById(R.id.showToTargetImg);
        this.U0 = (TextView) findViewById(R.id.showToTargetTxt);
        this.N0 = (DrawerLayout) findViewById(R.id.drawer_layout);
        this.D0 = (RelativeLayout) findViewById(R.id.loaderLayout);
        this.A0 = MediaPlayer.create(this, (int) R.raw.audio_error);
        this.t = (AutoFitTextureView) findViewById(R.id.autofittextureview);
        this.O0 = (ImageView) findViewById(R.id.helpBtn);
        this.t.f5665d = this.S / this.R;
        this.B0 = (RelativeLayout) findViewById(R.id.scanninglayout);
        this.E0 = (RelativeLayout) findViewById(R.id.arobjectfoundlayout);
        this.C0 = (RelativeLayout) findViewById(R.id.messageLayout);
        this.M0 = (ImageButton) findViewById(R.id.arGalleryButton);
        this.H0 = (ImageButton) findViewById(R.id.scanButton);
        this.L0 = (ImageButton) findViewById(R.id.toggleCameraVideo);
        this.T0 = (TextView) findViewById(R.id.toggleCameraVideoLabel);
        this.Q0 = (TextView) findViewById(R.id.videoTimerTxt);
        this.R0 = (TextView) findViewById(R.id.statusTxt);
        TextView textView = (TextView) findViewById(R.id.scanButtonTxt);
        this.S0 = (TextView) findViewById(R.id.arGalleryButtonTxt);
        this.r0 = (TextView) findViewById(R.id.leave_custom_user);
        this.G0 = (RelativeLayout) findViewById(R.id.cutom_user_layout);
        WindowManager windowManager = getWindowManager();
        Point point = new Point();
        windowManager.getDefaultDisplay().getSize(point);
        int i = point.x;
        int i2 = point.y;
        RelativeLayout.LayoutParams layoutParams = new RelativeLayout.LayoutParams(-2, -2);
        int i3 = i - (i - (i / 15));
        layoutParams.setMargins(i3, i2 - (i2 - (i2 / 12)), i3, i2 - (i2 - (i2 / 5)));
        this.F0.setLayoutParams(layoutParams);
        this.F0.setClipToOutline(true);
        this.L0.setOnClickListener(new View.OnClickListener() { // from class: c.e.b.h1
            @Override // android.view.View.OnClickListener
            public final void onClick(View view) {
                final ImageTrackingActivity imageTrackingActivity = ImageTrackingActivity.this;
                if (imageTrackingActivity.l0) {
                    Toast.makeText(imageTrackingActivity.D, "Recording OFF", 1).show();
                    imageTrackingActivity.T0.setText(imageTrackingActivity.getResources().getString(R.string.photo_video));
                    imageTrackingActivity.x();
                    return;
                }
                final Bitmap createBitmap = Bitmap.createBitmap(imageTrackingActivity.v.getArSceneView().getWidth(), imageTrackingActivity.v.getArSceneView().getHeight(), Bitmap.Config.ARGB_8888);
                PixelCopy.request(imageTrackingActivity.v.getArSceneView(), createBitmap, new PixelCopy.OnPixelCopyFinishedListener() { // from class: c.e.b.v1
                    @Override // android.view.PixelCopy.OnPixelCopyFinishedListener
                    public final void onPixelCopyFinished(int i4) {
                        String str;
                        ImageTrackingActivity imageTrackingActivity2 = ImageTrackingActivity.this;
                        Bitmap bitmap = createBitmap;
                        Objects.requireNonNull(imageTrackingActivity2);
                        if (i4 == 0) {
                            Log.d("ImageTrackingActivity", "bitmapReady");
                            Date date = new Date();
                            DateFormat.format("yyyy-MM-dd_hh:mm:ss", date);
                            try {
                                str = imageTrackingActivity2.getCacheDir().getAbsolutePath() + "/" + date + ".jpg";
                                FileOutputStream fileOutputStream = new FileOutputStream(new File(str));
                                bitmap.compress(Bitmap.CompressFormat.JPEG, 80, fileOutputStream);
                                fileOutputStream.flush();
                                fileOutputStream.close();
                            } catch (Throwable th) {
                                th.printStackTrace();
                                str = null;
                            }
                            bitmap.recycle();
                            if (str != null) {
                                imageTrackingActivity2.H(str, false);
                                return;
                            }
                            return;
                        }
                        Log.e("ImageTrackingActivity", "captureImage error");
                    }
                }, new Handler());
            }
        });
        this.L0.setOnLongClickListener(new View.OnLongClickListener() { // from class: c.e.b.x1
            @Override // android.view.View.OnLongClickListener
            public final boolean onLongClick(View view) {
                ImageTrackingActivity imageTrackingActivity = ImageTrackingActivity.this;
                if (imageTrackingActivity.l0) {
                    Toast.makeText(imageTrackingActivity.D, "Recording OFF", 1).show();
                    imageTrackingActivity.T0.setText(imageTrackingActivity.getResources().getString(R.string.photo_video));
                } else {
                    Toast.makeText(imageTrackingActivity.D, "Recording ON", 1).show();
                    imageTrackingActivity.T0.setText(imageTrackingActivity.getResources().getString(R.string.stop_recording));
                }
                imageTrackingActivity.x();
                return true;
            }
        });
        this.H0.setOnClickListener(new View.OnClickListener() { // from class: c.e.b.r1
            @Override // android.view.View.OnClickListener
            public final void onClick(View view) {
                final ImageTrackingActivity imageTrackingActivity = ImageTrackingActivity.this;
                boolean z = imageTrackingActivity.k0;
                if (z && imageTrackingActivity.M == 0) {
                    imageTrackingActivity.M();
                } else if (imageTrackingActivity.M == 2) {
                    if (!imageTrackingActivity.l0) {
                        new Handler().postDelayed(new Runnable() { // from class: c.e.b.w1
                            @Override // java.lang.Runnable
                            public final void run() {
                                ImageTrackingActivity imageTrackingActivity2 = ImageTrackingActivity.this;
                                int i4 = ImageTrackingActivity.r;
                                imageTrackingActivity2.M();
                            }
                        }, 500L);
                    }
                    imageTrackingActivity.B();
                } else if (z) {
                } else {
                    imageTrackingActivity.J();
                }
            }
        });
        this.M0.setOnClickListener(new View.OnClickListener() { // from class: c.e.b.t1
            @Override // android.view.View.OnClickListener
            public final void onClick(View view) {
                ImageTrackingActivity imageTrackingActivity = ImageTrackingActivity.this;
                Objects.requireNonNull(imageTrackingActivity);
                imageTrackingActivity.startActivity(new Intent(imageTrackingActivity.D, ARGalleryActivity.class));
            }
        });
        this.K0.setOnClickListener(new View.OnClickListener() { // from class: c.e.b.x0
            @Override // android.view.View.OnClickListener
            public final void onClick(View view) {
                ImageTrackingActivity imageTrackingActivity = ImageTrackingActivity.this;
                imageTrackingActivity.O0.setTag("2");
                if (imageTrackingActivity.l0) {
                    Toast.makeText(imageTrackingActivity.D, "Recording OFF", 1).show();
                    imageTrackingActivity.x();
                    return;
                }
                imageTrackingActivity.B();
            }
        });
        this.s = new yb(this, this.t, this, true);
        this.O0.setOnClickListener(new View.OnClickListener() { // from class: c.e.b.g2
            @Override // android.view.View.OnClickListener
            public final void onClick(View view) {
                ImageTrackingActivity imageTrackingActivity = ImageTrackingActivity.this;
                if (imageTrackingActivity.O0.getTag().equals("1")) {
                    imageTrackingActivity.startActivity(new Intent(imageTrackingActivity, HelpActivity.class));
                } else if (imageTrackingActivity.O0.getTag().equals("2")) {
                    imageTrackingActivity.startActivity(new Intent(imageTrackingActivity, Help2Activity.class));
                }
            }
        });
        this.I0.setOnClickListener(new View.OnClickListener() { // from class: c.e.b.u0
            @Override // android.view.View.OnClickListener
            public final void onClick(View view) {
                ImageTrackingActivity imageTrackingActivity = ImageTrackingActivity.this;
                imageTrackingActivity.B0.setVisibility(8);
                imageTrackingActivity.P();
            }
        });
        this.r0.setOnClickListener(new fc(this));
        if (!OpenCVLoader.initDebug()) {
            Log.d("ImageTrackingActivity", "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, this.Z0);
        } else {
            Log.d("ImageTrackingActivity", "OpenCV library found inside package. Using it!");
            this.Z0.onManagerConnected(0);
        }
        try {
            ActivityManager.MemoryInfo memoryInfo = new ActivityManager.MemoryInfo();
            ((ActivityManager) getSystemService("activity")).getMemoryInfo(memoryInfo);
            this.T = memoryInfo.totalMem / 1000000;
        } catch (Exception e2) {
            e2.printStackTrace();
            this.T = 4000L;
        }
        ArCoreApk.Availability checkAvailability = ArCoreApk.getInstance().checkAvailability(this);
        StringBuilder x = c.b.a.a.a.x("ARCoreSupported = ");
        x.append(checkAvailability.isSupported());
        Log.d("CheckARCoreSupport", x.toString());
        this.h0 = checkAvailability.isSupported() && (checkAvailability == ArCoreApk.Availability.SUPPORTED_INSTALLED || checkAvailability == ArCoreApk.Availability.SUPPORTED_APK_TOO_OLD);
        this.y0 = new Handler();
        this.z0 = new Handler();
        if (this.C.f4871a.getString("arContentServer", "").isEmpty()) {
            this.D0.setVisibility(0);
            J();
        }
        NavigationView navigationView = (NavigationView) findViewById(R.id.nav_view);
        navigationView.setNavigationItemSelectedListener(new NavigationView.OnNavigationItemSelectedListener() { // from class: c.e.b.o1
            @Override // com.google.android.material.navigation.NavigationView.OnNavigationItemSelectedListener
            public final boolean onNavigationItemSelected(MenuItem menuItem) {
                ImageTrackingActivity imageTrackingActivity = ImageTrackingActivity.this;
                Objects.requireNonNull(imageTrackingActivity);
                new Handler().postDelayed(new o2(imageTrackingActivity, menuItem.getItemId()), 200L);
                imageTrackingActivity.N0.c(false);
                return true;
            }
        });
        Button button = (Button) navigationView.getHeaderView(0).findViewById(R.id.nav_visitunitear);
        this.W0 = button;
        button.setOnClickListener(new View.OnClickListener() { // from class: c.e.b.d2
            @Override // android.view.View.OnClickListener
            public final void onClick(View view) {
                ImageTrackingActivity imageTrackingActivity = ImageTrackingActivity.this;
                Objects.requireNonNull(imageTrackingActivity);
                new Handler().postDelayed(new o2(imageTrackingActivity, R.id.nav_visitunitear), 200L);
                imageTrackingActivity.N0.c(false);
            }
        });
        ImageButton imageButton = (ImageButton) findViewById(R.id.navDrawerButton);
        this.J0 = imageButton;
        imageButton.setOnClickListener(new View.OnClickListener() { // from class: c.e.b.d1
            @Override // android.view.View.OnClickListener
            public final void onClick(View view) {
                ImageTrackingActivity imageTrackingActivity = ImageTrackingActivity.this;
                DrawerLayout drawerLayout = imageTrackingActivity.N0;
                View d2 = drawerLayout.d(8388611);
                if (d2 != null ? drawerLayout.l(d2) : false) {
                    imageTrackingActivity.N0.c(false);
                    return;
                }
                DrawerLayout drawerLayout2 = imageTrackingActivity.N0;
                View d3 = drawerLayout2.d(8388611);
                if (d3 != null) {
                    drawerLayout2.o(d3, true);
                    return;
                }
                StringBuilder x2 = c.b.a.a.a.x("No drawer view found with gravity ");
                x2.append(DrawerLayout.i(8388611));
                throw new IllegalArgumentException(x2.toString());
            }
        });
        SimpleFragment simpleFragment = (SimpleFragment) m().H(R.id.simple_sceneform_fragment);
        this.v = simpleFragment;
        simpleFragment.getArSceneView().getScene().addOnUpdateListener(this);
        this.v.getArSceneView().getScene().addOnPeekTouchListener(this);
        SimpleSceneView arSceneView = this.v.getArSceneView();
        this.w = arSceneView;
        arSceneView.getScene().getCamera().setVerticalFovDegrees(60.0f);
        Node node = new Node();
        this.x = node;
        node.setWorldPosition(new Vector3(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, -5.2f));
        this.x.setWorldRotation(new Quaternion(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, -0.707f, 0.707f));
        this.x.setParent(this.w.getScene().getCamera());
        this.u = new ExternalTexture();
        Material.builder().setSource(this, R.raw.augmented_video_material).build().thenAccept(new Consumer() { // from class: c.e.b.k1
            @Override // java.util.function.Consumer
            public final void accept(Object obj) {
                ImageTrackingActivity imageTrackingActivity = ImageTrackingActivity.this;
                Objects.requireNonNull(imageTrackingActivity);
                imageTrackingActivity.x.setRenderable(ShapeFactory.makeCube(new Vector3(6.0f, 4.5f, 0.1f), Vector3.zero(), (Material) obj));
                Renderable renderable = imageTrackingActivity.x.getRenderable();
                Objects.requireNonNull(renderable);
                renderable.getMaterial().setExternalTexture("videoTexture", imageTrackingActivity.u);
            }
        }).exceptionally(new Function() { // from class: c.e.b.m2
            @Override // java.util.function.Function
            public final Object apply(Object obj) {
                Throwable th = (Throwable) obj;
                Objects.requireNonNull(ImageTrackingActivity.this);
                Log.e("ImageTrackingActivity", "Unable to load camera renderable");
                return null;
            }
        });
        w(this.w);
        this.z = new p();
        SensorManager sensorManager = (SensorManager) getSystemService("sensor");
        this.E = sensorManager;
        this.F = sensorManager.getDefaultSensor(11);
        if (Build.VERSION.SDK_INT >= 26 && (intent = getIntent()) != null && intent.getData() != null) {
            b.v.u.c.z(this);
            L("Loading data.. Please wait", 2000);
            String str = new String(Base64.getUrlDecoder().decode(intent.getData().getQuery()));
            this.Y0 = str;
            c.e.b.p000if.d dVar = this.C;
            dVar.f4872b.putString("custom_user", str);
            dVar.f4872b.apply();
            J();
        }
        this.Y0 = this.C.f4871a.getString("custom_user", "");
        if (!this.h0) {
            this.i0 = true;
        }
        this.u0.patternDetectorInitJNI();
    }

    @Override // b.b.c.h, b.q.b.d, android.app.Activity
    public void onDestroy() {
        super.onDestroy();
        try {
            this.s0.cancel();
            this.s.b();
        } catch (Exception e2) {
            Log.e("ImageTrackingActivity", "onDestroy " + e2);
        }
        if (this.F != null) {
            this.E.unregisterListener(this);
        }
        c.e.b.p000if.f.a(this.D, 43200);
        Log.d("ImageTrackingActivity", "Clear cached 3D models");
        ((NotificationManager) this.D.getSystemService("notification")).cancel(10);
    }

    @Override // b.q.b.d, android.app.Activity, android.content.ComponentCallbacks
    public void onLowMemory() {
        super.onLowMemory();
        c.e.b.p000if.f.a(this.D, 60);
        Log.d("ImageTrackingActivity", "Clear cached 3D models");
    }

    @Override // b.q.b.d, android.app.Activity
    public void onNewIntent(Intent intent) {
        super.onNewIntent(intent);
        if (Build.VERSION.SDK_INT < 26 || intent == null || intent.getData() == null) {
            return;
        }
        b.v.u.c.z(this);
        L("Loading data.. Please wait", 2000);
        String str = new String(Base64.getUrlDecoder().decode(intent.getData().getQuery()));
        this.Y0 = str;
        c.e.b.p000if.d dVar = this.C;
        dVar.f4872b.putString("custom_user", str);
        dVar.f4872b.apply();
        J();
    }

    @Override // b.q.b.d, android.app.Activity
    public void onPause() {
        super.onPause();
        this.f0 = true;
        if (this.l0) {
            this.B.f4904c.pause();
        }
        if (this.F != null) {
            this.E.unregisterListener(this);
        }
        if (this.j0) {
            N();
            this.d0 = true;
            A();
        }
        try {
            this.y.setEnabled(false);
            hd hdVar = this.A;
            if (hdVar != null) {
                hdVar.r();
            }
            this.s.b();
        } catch (Exception e2) {
            Log.e("ImageTrackingActivity", "onPause " + e2);
        }
        this.e0 = false;
    }

    @Override // com.google.ar.sceneform.Scene.OnPeekTouchListener
    public void onPeekTouch(HitTestResult hitTestResult, MotionEvent motionEvent) {
        for (Node node : this.y.getChildren()) {
            if (node instanceof SimpleTransformableNode) {
                SimpleTransformableNode simpleTransformableNode = (SimpleTransformableNode) node;
                if (simpleTransformableNode.isSelected()) {
                    this.z.a(motionEvent, simpleTransformableNode);
                }
            }
        }
    }

    @Override // b.q.b.d, android.app.Activity
    public void onResume() {
        super.onResume();
        c.e.b.p000if.d dVar = this.C;
        if (dVar != null) {
            if (!dVar.f4871a.getString("arContentServer", "").isEmpty() && this.Y0.length() == 0) {
                if (System.currentTimeMillis() - Long.valueOf(this.C.f4871a.getLong("last_updated", 0L)).longValue() > 43200000) {
                    this.D0.setVisibility(0);
                    J();
                } else {
                    E();
                }
            } else {
                E();
            }
            if (this.Y0.length() != 0) {
                this.G0.setVisibility(0);
            }
        }
        try {
            this.X = false;
            if (this.C.f4871a.getString("fromHistory", "").length() != 0) {
                hd hdVar = new hd(this.C.f4871a.getString("fromHistory", ""), ac.f4547a.f4549c, this.y, this.v, this, this);
                this.A = hdVar;
                hdVar.u(new hd.g() { // from class: c.e.b.c2
                    @Override // c.e.b.hd.g
                    public final void a(final String str) {
                        final ImageTrackingActivity imageTrackingActivity = ImageTrackingActivity.this;
                        Objects.requireNonNull(imageTrackingActivity);
                        if (str.length() != 0) {
                            imageTrackingActivity.runOnUiThread(new Runnable() { // from class: c.e.b.s0
                                @Override // java.lang.Runnable
                                public final void run() {
                                    ImageTrackingActivity.this.L(str, 5000);
                                }
                            });
                        }
                    }
                });
                c.e.b.p000if.d dVar2 = this.C;
                dVar2.f4872b.putString("fromHistory", "");
                dVar2.f4872b.apply();
                new Handler().postDelayed(new Runnable() { // from class: c.e.b.p1
                    @Override // java.lang.Runnable
                    public final void run() {
                        ImageTrackingActivity imageTrackingActivity = ImageTrackingActivity.this;
                        imageTrackingActivity.G();
                        imageTrackingActivity.P();
                    }
                }, 1000L);
            }
            if (this.C.f4871a.getString("arGalleryFile", "").length() != 0) {
                try {
                    String string = this.C.f4871a.getString("arGalleryFile", "");
                    c.e.b.p000if.d dVar3 = this.C;
                    dVar3.f4872b.putString("arGalleryFile", "");
                    dVar3.f4872b.apply();
                    Q(string, this.C.f4871a.getString("arGalleryFileId", ""));
                } catch (Exception e2) {
                    Log.e("ImageTrackingActivity", "onResume " + e2);
                }
            } else {
                if (this.f0) {
                    this.f0 = false;
                    if (this.e0) {
                        this.A.v();
                    }
                    if (this.t.isAvailable()) {
                        this.s.e(yb.f5442e, yb.f5443f);
                    }
                }
                if (this.F != null) {
                    SensorManager sensorManager = this.E;
                    sensorManager.registerListener(this, sensorManager.getDefaultSensor(11), 1);
                }
            }
            if (this.M == 2 && !this.e0 && this.a0 && !this.g0) {
                this.P0.setVisibility(0);
                this.U0.setVisibility(0);
            }
            if (this.M == 2) {
                this.g0 = false;
            }
            this.n0 = this.C.f4871a.getBoolean("Gyro", false);
            Log.d("ImageTrackingActivity", "Enables regular immersive mode");
            getWindow().getDecorView().setSystemUiVisibility(4871);
            if (this.l0) {
                this.B.f4904c.resume();
            }
        } catch (Exception e3) {
            Log.e("ImageTrackingActivity", "OnResume " + e3);
        }
    }

    @Override // android.hardware.SensorEventListener
    public void onSensorChanged(SensorEvent sensorEvent) {
        if (sensorEvent.sensor.getType() == 11) {
            if (this.e0 && !this.g0) {
                if (this.G != null) {
                    this.G = null;
                    this.w.getScene().getCamera().setLocalRotation(Quaternion.identity());
                }
            } else if (!this.g0 && this.n0) {
                Node node = this.y;
                node.setLocalRotation(Quaternion.slerp(node.getLocalRotation(), Quaternion.identity(), 0.1f));
                this.w.getScene().getCamera().setLocalRotation(Quaternion.slerp(this.w.getScene().getCamera().getLocalRotation(), Quaternion.identity(), 0.1f));
                this.w.getScene().getCamera().setWorldPosition(Vector3.lerp(this.w.getScene().getCamera().getWorldPosition(), new Vector3(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 2.0f), 0.1f));
            } else {
                float[] fArr = new float[16];
                this.P = fArr;
                SensorManager.getRotationMatrixFromVector(fArr, sensorEvent.values);
                float[] fArr2 = this.P;
                Vector3 vector3 = new Vector3(-fArr2[2], -fArr2[6], -fArr2[10]);
                float[] fArr3 = this.P;
                Quaternion lookRotation = Quaternion.lookRotation(vector3, new Vector3(fArr3[1], fArr3[5], fArr3[9]));
                if (this.G == null) {
                    this.G = lookRotation.inverted();
                }
                this.w.getScene().getCamera().setLocalRotation(Quaternion.slerp(this.w.getScene().getCamera().getLocalRotation(), Quaternion.multiply(this.G, lookRotation), 0.2f));
            }
        }
    }

    @Override // b.b.c.h, b.q.b.d, android.app.Activity
    public void onStop() {
        super.onStop();
        AppUpdateManager appUpdateManager = this.X0;
        if (appUpdateManager != null) {
            appUpdateManager.unregisterListener(this.a1);
        }
    }

    @Override // com.google.ar.sceneform.Scene.OnUpdateListener
    public void onUpdate(FrameTime frameTime) {
    }

    @Override // android.app.Activity, android.view.Window.Callback
    public void onWindowFocusChanged(boolean z) {
        super.onWindowFocusChanged(z);
        if (z) {
            Log.d("ImageTrackingActivity", "Enables regular immersive mode");
            getWindow().getDecorView().setSystemUiVisibility(4871);
        }
    }

    public final void w(SceneView sceneView) {
        Node node = new Node();
        this.y = node;
        node.setParent(sceneView.getScene());
        this.y.setLocalScale(new Vector3(4.0f, 4.0f, 4.0f));
        Material.builder().setSource(this, R.raw.sceneform_opaque_colored_material).build().thenAccept((Consumer<? super Material>) b1.f4559a).exceptionally(new Function() { // from class: c.e.b.b2
            @Override // java.util.function.Function
            public final Object apply(Object obj) {
                Throwable th = (Throwable) obj;
                Objects.requireNonNull(ImageTrackingActivity.this);
                Log.e("ImageTrackingActivity", "Unable to load camera renderable");
                return null;
            }
        });
    }

    public final void x() {
        new Handler().postDelayed(new Runnable() { // from class: c.e.b.g1
            @Override // java.lang.Runnable
            public final void run() {
                ImageTrackingActivity imageTrackingActivity = ImageTrackingActivity.this;
                imageTrackingActivity.Q0.setVisibility(0);
                imageTrackingActivity.O = 0;
                imageTrackingActivity.N = 0;
                imageTrackingActivity.Q0.setText(String.format(Locale.ENGLISH, "%02d:%02d", 0, Integer.valueOf(imageTrackingActivity.N)));
                if (imageTrackingActivity.B == null) {
                    c.e.b.p000if.q qVar = new c.e.b.p000if.q(imageTrackingActivity.D);
                    imageTrackingActivity.B = qVar;
                    qVar.f4906e = imageTrackingActivity.v.getArSceneView();
                    int i = imageTrackingActivity.getResources().getConfiguration().orientation;
                    if (imageTrackingActivity.R > 1000.0f) {
                        imageTrackingActivity.B.d(6, i);
                    } else {
                        imageTrackingActivity.B.d(5, i);
                    }
                }
                if (!imageTrackingActivity.l0) {
                    boolean b2 = imageTrackingActivity.B.b();
                    imageTrackingActivity.l0 = b2;
                    if (b2) {
                        Timer timer = imageTrackingActivity.t0;
                        if (timer != null) {
                            timer.cancel();
                        }
                        imageTrackingActivity.O = 0;
                        imageTrackingActivity.N = -1;
                        Timer timer2 = new Timer();
                        imageTrackingActivity.t0 = timer2;
                        timer2.scheduleAtFixedRate(new hc(imageTrackingActivity), 0L, 1000L);
                        imageTrackingActivity.C(true);
                        return;
                    }
                    return;
                }
                imageTrackingActivity.Q0.setVisibility(8);
                imageTrackingActivity.l0 = imageTrackingActivity.B.b();
                imageTrackingActivity.C(false);
                imageTrackingActivity.N = 0;
                imageTrackingActivity.O = 0;
                Timer timer3 = imageTrackingActivity.t0;
                if (timer3 != null) {
                    timer3.cancel();
                }
                String path = imageTrackingActivity.B.i.getPath();
                imageTrackingActivity.K = path;
                imageTrackingActivity.H(path, true);
            }
        }, 300L);
    }

    public final void y() {
        this.L = 0;
        if (this.C.f4871a.getBoolean("force_update", false)) {
            this.L = 1;
        }
        this.X0.getAppUpdateInfo().addOnSuccessListener(new OnSuccessListener() { // from class: c.e.b.k2
            @Override // com.google.android.play.core.tasks.OnSuccessListener
            public final void onSuccess(Object obj) {
                ImageTrackingActivity imageTrackingActivity = ImageTrackingActivity.this;
                AppUpdateInfo appUpdateInfo = (AppUpdateInfo) obj;
                if (Integer.valueOf(imageTrackingActivity.C.f4871a.getInt("update_req", 0)).intValue() == appUpdateInfo.availableVersionCode() && imageTrackingActivity.L == 0) {
                    return;
                }
                Log.e("AVAILABLE_VERSION_CODE", appUpdateInfo.availableVersionCode() + "");
                if (appUpdateInfo.updateAvailability() == 2 && appUpdateInfo.isUpdateTypeAllowed(imageTrackingActivity.L)) {
                    try {
                        imageTrackingActivity.X0.startUpdateFlowForResult(appUpdateInfo, imageTrackingActivity.L, imageTrackingActivity, 1);
                        c.e.b.p000if.d dVar = imageTrackingActivity.C;
                        dVar.f4872b.putInt("update_req", appUpdateInfo.availableVersionCode());
                        dVar.f4872b.apply();
                    } catch (IntentSender.SendIntentException unused) {
                        Log.e("AVAILABLE_VERSION_CODE", appUpdateInfo.availableVersionCode() + "");
                    }
                }
            }
        });
        this.X0.registerListener(this.a1);
    }

    public final Mat z(int i, int i2) {
        Objects.requireNonNull(this.s);
        int i3 = yb.f5438a;
        if (i3 == 90 || i3 == 270) {
            Mat mat = new Mat(i, i2, CvType.CV_8UC1);
            if (i3 == 90) {
                Core.rotate(this.J, mat, 0);
            } else {
                Core.rotate(this.J, mat, 2);
            }
            return mat;
        } else if (i3 == 180) {
            Mat mat2 = new Mat(i2, i, CvType.CV_8UC1);
            Core.rotate(this.J, mat2, 1);
            return mat2;
        } else {
            Mat mat3 = new Mat(i2, i, CvType.CV_8UC1);
            this.J.copyTo(mat3);
            return mat3;
        }
    }
}