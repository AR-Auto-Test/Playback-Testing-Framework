package com.ibosoninnov.unitear;

import android.app.Dialog;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.drawable.ColorDrawable;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.media.MediaPlayer;
import android.net.Uri;
import android.opengl.Matrix;
import android.os.Bundle;
import android.os.Handler;
import android.provider.Settings;
import android.text.format.DateFormat;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.MotionEvent;
import android.view.PixelCopy;
import android.view.View;
import android.view.ViewGroup;
import android.view.animation.AnimationUtils;
import android.view.animation.TranslateAnimation;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.RelativeLayout;
import android.widget.TextView;
import android.widget.Toast;
import androidx.cardview.widget.CardView;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.recyclerview.widget.RecyclerView;
import b.b.c.h;
import c.e.b.ac;
import c.e.b.ba;
import c.e.b.ec;
import c.e.b.ef.c;
import c.e.b.ef.f;
import c.e.b.gf.b;
import c.e.b.hf.d;
import c.e.b.p000if.j;
import c.e.b.p000if.l;
import c.e.b.p000if.p;
import c.e.b.p000if.q;
import c.e.b.rd;
import c.e.b.sb;
import c.e.b.vc;
import c.e.b.yd;
import com.google.android.material.bottomsheet.BottomSheetDialog;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.google.ar.core.Anchor;
import com.google.ar.core.Camera;
import com.google.ar.core.Config;
import com.google.ar.core.Frame;
import com.google.ar.core.HitResult;
import com.google.ar.core.InstallActivity;
import com.google.ar.core.Plane;
import com.google.ar.core.Session;
import com.google.ar.core.Trackable;
import com.google.ar.core.TrackingState;
import com.google.ar.core.exceptions.UnavailableApkTooOldException;
import com.google.ar.core.exceptions.UnavailableArcoreNotInstalledException;
import com.google.ar.core.exceptions.UnavailableDeviceNotCompatibleException;
import com.google.ar.core.exceptions.UnavailableSdkTooOldException;
import com.google.ar.sceneform.AnchorNode;
import com.google.ar.sceneform.ArSceneView;
import com.google.ar.sceneform.FrameTime;
import com.google.ar.sceneform.HitTestResult;
import com.google.ar.sceneform.Node;
import com.google.ar.sceneform.Scene;
import com.google.ar.sceneform.math.Quaternion;
import com.google.ar.sceneform.math.Vector3;
import com.google.ar.sceneform.rendering.ExternalTexture;
import com.google.ar.sceneform.rendering.Material;
import com.google.ar.sceneform.rendering.ModelRenderable;
import com.google.ar.sceneform.rendering.ViewRenderable;
import com.google.ar.sceneform.ux.ArFragment;
import com.google.ar.sceneform.ux.SimpleTransformableNode;
import com.google.firebase.crashlytics.internal.settings.SettingsJsonConstants;
import com.ibosoninnov.unitear.ARCoreSceneformActivity;
import com.ibosoninnov.unitear.R;
import com.ibosoninnov.unitear.activities.Help2Activity;
import f.u;
import java.io.File;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.Date;
import java.util.Iterator;
import java.util.Objects;
import java.util.Timer;
import java.util.function.Consumer;
import java.util.function.Function;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

/* loaded from: classes2.dex */
public class ARCoreSceneformActivity extends h implements c.e.b.gf.a, b, Scene.OnUpdateListener, Scene.OnPeekTouchListener {
    public static SensorManager r;
    public float A;
    public float B;
    public Context D;
    public boolean F;
    public Sensor G;
    public int H;
    public int I;
    public yd N;
    public vc O;
    public ImageButton Q;
    public ImageButton R;
    public ImageButton S;
    public ImageButton T;
    public ImageButton U;
    public ImageView V;
    public ImageView W;
    public RelativeLayout X;
    public RelativeLayout Y;
    public LinearLayout Z;
    public TextView a0;
    public TextView b0;
    public ConstraintLayout c0;
    public CardView d0;
    public CardView e0;
    public boolean h0;
    public String i0;
    public Timer j0;
    public l k0;
    public q l0;
    public BottomSheetDialog m0;
    public RecyclerView n0;
    public RecyclerView o0;
    public c r0;
    public ArFragment s;
    public f s0;
    public ArSceneView t;
    public Node u;
    public SimpleTransformableNode v;
    public ProgressBar v0;
    public p w;
    public TextView w0;
    public String x;
    public Dialog x0;
    public final ArrayList<Anchor> y = new ArrayList<>();
    public final float[] z = new float[16];
    public boolean C = false;
    public boolean E = false;
    public int J = 0;
    public String K = "";
    public String L = null;
    public String M = "";
    public boolean P = false;
    public int f0 = 0;
    public int g0 = 0;
    public ArrayList<c.e.b.hf.a> p0 = null;
    public ArrayList<d> q0 = null;
    public ArrayList<c.e.b.hf.a> t0 = new ArrayList<>();
    public ArrayList<String> u0 = new ArrayList<>();
    public String y0 = "";
    public SensorEventListener z0 = new a();

    /* loaded from: classes2.dex */
    public class a implements SensorEventListener {
        public a() {
        }

        @Override // android.hardware.SensorEventListener
        public void onAccuracyChanged(Sensor sensor, int i) {
        }

        @Override // android.hardware.SensorEventListener
        public void onSensorChanged(SensorEvent sensorEvent) {
            ARCoreSceneformActivity aRCoreSceneformActivity = ARCoreSceneformActivity.this;
            float[] fArr = sensorEvent.values;
            Objects.requireNonNull(aRCoreSceneformActivity);
        }
    }

    public final void A(boolean z, boolean z2) {
        Log.d("ARCoreSceneformActivity", "TapToScanUI " + z);
        if (z) {
            this.Z.setVisibility(8);
            this.c0.setVisibility(8);
            this.X.setVisibility(0);
            this.V.setAnimation(AnimationUtils.loadAnimation(this.D, R.anim.scanfloor_anim));
            C(getResources().getString(R.string.detecting_surface), true, 0);
            return;
        }
        if (z2) {
            this.E = true;
            this.d0.setVisibility(8);
            new Handler().postDelayed(new Runnable() { // from class: c.e.b.c
                @Override // java.lang.Runnable
                public final void run() {
                    ARCoreSceneformActivity aRCoreSceneformActivity = ARCoreSceneformActivity.this;
                    aRCoreSceneformActivity.Z.setVisibility(0);
                    aRCoreSceneformActivity.e0.setVisibility(8);
                }
            }, 2000L);
        }
        if (this.L != null) {
            this.c0.setVisibility(0);
            this.Z.setVisibility(0);
        }
        if (this.L == null) {
            this.c0.setVisibility(8);
        }
        this.R.setVisibility(0);
        this.X.setVisibility(8);
        this.V.clearAnimation();
        C(getResources().getString(R.string.surface_found), false, 2000);
    }

    public void B() {
        String str;
        Iterator<d> it = this.q0.iterator();
        while (it.hasNext()) {
            it.next().isSelected = false;
        }
        if (this.q0.size() > 0) {
            this.q0.get(0).isSelected = true;
            str = this.q0.get(0).name;
        } else {
            str = "";
        }
        this.t0.clear();
        Iterator<c.e.b.hf.a> it2 = this.p0.iterator();
        while (it2.hasNext()) {
            c.e.b.hf.a next = it2.next();
            if (next.category.equals(str)) {
                this.t0.add(next);
            }
            if (new File(getCacheDir(), c.b.a.a.a.v(c.b.a.a.a.x("/assets/models/"), next.id, ".glb")).exists()) {
                next.downloadStatus = 101;
            }
        }
        this.r0 = new c(this.q0, this);
        this.m0 = new BottomSheetDialog(this);
        View inflate = LayoutInflater.from(this).inflate(R.layout.ar_gallery_bottom_sheet, (ViewGroup) null);
        this.n0 = (RecyclerView) inflate.findViewById(R.id.category_rec_view);
        this.o0 = (RecyclerView) inflate.findViewById(R.id.thumbnails_rec_view);
        this.n0.setAdapter(this.r0);
        f fVar = new f(this.t0, this);
        this.s0 = fVar;
        this.o0.setAdapter(fVar);
        this.m0.setContentView(inflate);
        ((ConstraintLayout) inflate.findViewById(R.id.dismiss)).setOnClickListener(new View.OnClickListener() { // from class: c.e.b.f
            @Override // android.view.View.OnClickListener
            public final void onClick(View view) {
                ARCoreSceneformActivity.this.m0.dismiss();
            }
        });
        this.m0.setOnCancelListener(new DialogInterface.OnCancelListener() { // from class: c.e.b.u
            @Override // android.content.DialogInterface.OnCancelListener
            public final void onCancel(DialogInterface dialogInterface) {
                ARCoreSceneformActivity.this.c0.setVisibility(0);
            }
        });
        this.m0.show();
        this.c0.setVisibility(4);
    }

    public final void C(String str, boolean z, int i) {
        Log.d("ARCoreSceneformActivity", "showTopMessage - " + str);
        ((TextView) findViewById(R.id.msgTxt)).setText(str);
        if (z) {
            this.Y.setVisibility(0);
        } else {
            new Handler().postDelayed(new Runnable() { // from class: c.e.b.p
                @Override // java.lang.Runnable
                public final void run() {
                    ARCoreSceneformActivity.this.Y.setVisibility(8);
                }
            }, i);
        }
    }

    @Override // c.e.b.gf.a
    public void c(c.e.b.hf.a aVar) {
        this.m0.dismiss();
        StringBuilder sb = new StringBuilder();
        sb.append(this.D.getCacheDir());
        sb.append("/assets/models/");
        String v = c.b.a.a.a.v(sb, aVar.id, ".glb");
        this.c0.setVisibility(0);
        this.J = 0;
        yd ydVar = this.N;
        ydVar.x = true;
        for (Node node : ydVar.f5453d.getChildren()) {
            node.setParent(null);
        }
        x(v);
    }

    @Override // c.e.b.gf.a
    public void d(c.e.b.hf.a aVar) {
        File file = new File(getCacheDir(), "assets/models");
        if (!file.exists()) {
            file.mkdir();
        }
        if (new File(this.D.getCacheDir(), c.b.a.a.a.v(c.b.a.a.a.x("assets/models/"), aVar.id, ".glb")).exists()) {
            f(aVar.id, 101, false, "");
        } else {
            new j(this, aVar.id, false).execute(aVar.file_loc);
        }
    }

    @Override // c.e.b.gf.b
    public void f(String str, int i, boolean z, String str2) {
        try {
            if (z) {
                if (this.x0.isShowing()) {
                    this.v0.setProgress(i);
                    this.w0.setText(str2);
                    if (i == 101) {
                        this.x0.dismiss();
                        String str3 = this.D.getCacheDir().getPath() + "/assets/models/" + str + ".glb";
                        Log.d("ARCoreSceneformActivity", "file path --  " + str3);
                        x(str3);
                    }
                }
            } else {
                for (int i2 = 0; i2 <= this.t0.size(); i2++) {
                    if (this.t0.get(i2).id.equals(str)) {
                        this.t0.get(i2).downloadStatus = i;
                        this.s0.notifyItemChanged(i2);
                        break;
                    }
                }
            }
        } catch (Exception e2) {
            e2.printStackTrace();
        }
    }

    @Override // c.e.b.gf.a
    public void h(String str) {
        for (int i = 0; i < this.q0.size(); i++) {
            this.q0.get(i).isSelected = this.q0.get(i).name.equals(str);
        }
        this.r0.notifyDataSetChanged();
        this.t0.clear();
        Iterator<c.e.b.hf.a> it = this.p0.iterator();
        while (it.hasNext()) {
            c.e.b.hf.a next = it.next();
            if (next.category.equals(str)) {
                this.t0.add(next);
            }
            if (new File(getCacheDir(), c.b.a.a.a.v(c.b.a.a.a.x("models/"), next.id, ".glb")).exists()) {
                next.downloadStatus = 101;
            }
        }
        f fVar = new f(this.t0, this);
        this.s0 = fVar;
        this.o0.setAdapter(fVar);
    }

    @Override // androidx.activity.ComponentActivity, android.app.Activity
    public void onBackPressed() {
        this.N.w();
        this.f41f.b();
    }

    @Override // b.b.c.h, b.q.b.d, androidx.activity.ComponentActivity, b.j.b.e, android.app.Activity
    public void onCreate(Bundle bundle) {
        String str;
        boolean z;
        super.onCreate(bundle);
        requestWindowFeature(1);
        getWindow().setFlags(1024, 1024);
        getWindow().addFlags(128);
        getWindow().addFlags(1536);
        setContentView(R.layout.activity_arcore_sceneform);
        this.c0 = (ConstraintLayout) findViewById(R.id.swipeLayout);
        this.W = (ImageView) findViewById(R.id.uparrow);
        this.c0.setVisibility(8);
        this.Z = (LinearLayout) findViewById(R.id.arobjectfoundlayout);
        this.X = (RelativeLayout) findViewById(R.id.scanTheFloorLayout);
        this.V = (ImageView) findViewById(R.id.scanFloorImg);
        this.Y = (RelativeLayout) findViewById(R.id.messageLayout);
        this.d0 = (CardView) findViewById(R.id.aimYourDeviceLayout);
        this.e0 = (CardView) findViewById(R.id.foundSurfaceLayout);
        this.Q = (ImageButton) findViewById(R.id.closeARContentButton);
        this.T = (ImageButton) findViewById(R.id.reloadbtn);
        this.U = (ImageButton) findViewById(R.id.helpimg);
        this.W.setOnClickListener(new View.OnClickListener() { // from class: c.e.b.h
            @Override // android.view.View.OnClickListener
            public final void onClick(View view) {
                ARCoreSceneformActivity.this.B();
            }
        });
        this.Q.setOnClickListener(new View.OnClickListener() { // from class: c.e.b.t
            @Override // android.view.View.OnClickListener
            public final void onClick(View view) {
                ARCoreSceneformActivity aRCoreSceneformActivity = ARCoreSceneformActivity.this;
                if (aRCoreSceneformActivity.h0) {
                    Toast.makeText(aRCoreSceneformActivity.D, "Recording OFF", 1).show();
                    aRCoreSceneformActivity.v();
                    return;
                }
                aRCoreSceneformActivity.u.setParent(null);
                aRCoreSceneformActivity.finish();
            }
        });
        this.U.setOnClickListener(new View.OnClickListener() { // from class: c.e.b.o
            @Override // android.view.View.OnClickListener
            public final void onClick(View view) {
                ARCoreSceneformActivity aRCoreSceneformActivity = ARCoreSceneformActivity.this;
                Objects.requireNonNull(aRCoreSceneformActivity);
                aRCoreSceneformActivity.startActivity(new Intent(aRCoreSceneformActivity, Help2Activity.class));
            }
        });
        this.T.setOnClickListener(new View.OnClickListener() { // from class: c.e.b.w
            @Override // android.view.View.OnClickListener
            public final void onClick(View view) {
                ARCoreSceneformActivity aRCoreSceneformActivity = ARCoreSceneformActivity.this;
                Objects.requireNonNull(aRCoreSceneformActivity);
                Log.d("ARCoreSceneformActivity", "reload scene");
                yd ydVar = aRCoreSceneformActivity.N;
                Iterator<Node> it = ydVar.o.iterator();
                while (it.hasNext()) {
                    Node next = it.next();
                    if (next.getName().equals("playPauseButton")) {
                        try {
                            ((ImageView) ((ViewRenderable) next.getRenderable()).getView().findViewById(R.id.img_loader_view)).setImageResource(R.drawable.play);
                        } catch (Exception e2) {
                            Log.e("LoaderARContentSceneformARCore", e2.toString());
                        }
                    }
                }
                ydVar.q();
                aRCoreSceneformActivity.A(true, false);
                aRCoreSceneformActivity.C = false;
                aRCoreSceneformActivity.u.setParent(null);
                aRCoreSceneformActivity.v.setParent(null);
                aRCoreSceneformActivity.v.setWorldPosition(new Vector3());
                aRCoreSceneformActivity.v.setWorldRotation(new Quaternion());
                aRCoreSceneformActivity.v.setLocalScale(new Vector3(1.1f, 1.1f, 1.1f));
                aRCoreSceneformActivity.v.getScaleController().setMinScale(0.1f);
                aRCoreSceneformActivity.v.getScaleController().setMaxScale(3.5f);
            }
        });
        ImageButton imageButton = (ImageButton) findViewById(R.id.groundplaneButton);
        this.R = imageButton;
        imageButton.setOnClickListener(new View.OnClickListener() { // from class: c.e.b.b
            @Override // android.view.View.OnClickListener
            public final void onClick(View view) {
                ARCoreSceneformActivity aRCoreSceneformActivity = ARCoreSceneformActivity.this;
                if (aRCoreSceneformActivity.h0) {
                    Toast.makeText(aRCoreSceneformActivity.D, "Recording OFF", 1).show();
                    aRCoreSceneformActivity.v();
                    return;
                }
                aRCoreSceneformActivity.u.setParent(null);
                aRCoreSceneformActivity.finish();
            }
        });
        this.S = (ImageButton) findViewById(R.id.photoVideoSwitch);
        this.b0 = (TextView) findViewById(R.id.takePhotoVideo);
        this.S.setOnClickListener(new View.OnClickListener() { // from class: c.e.b.m
            @Override // android.view.View.OnClickListener
            public final void onClick(View view) {
                final ARCoreSceneformActivity aRCoreSceneformActivity = ARCoreSceneformActivity.this;
                if (aRCoreSceneformActivity.h0) {
                    Toast.makeText(aRCoreSceneformActivity.D, "Recording OFF", 1).show();
                    aRCoreSceneformActivity.b0.setText(aRCoreSceneformActivity.getResources().getString(R.string.photo_video));
                    aRCoreSceneformActivity.v();
                    return;
                }
                final Bitmap createBitmap = Bitmap.createBitmap(aRCoreSceneformActivity.s.getArSceneView().getWidth(), aRCoreSceneformActivity.s.getArSceneView().getHeight(), Bitmap.Config.ARGB_8888);
                PixelCopy.request(aRCoreSceneformActivity.s.getArSceneView(), createBitmap, new PixelCopy.OnPixelCopyFinishedListener() { // from class: c.e.b.e
                    @Override // android.view.PixelCopy.OnPixelCopyFinishedListener
                    public final void onPixelCopyFinished(int i) {
                        String str2;
                        ARCoreSceneformActivity aRCoreSceneformActivity2 = ARCoreSceneformActivity.this;
                        Bitmap bitmap = createBitmap;
                        Objects.requireNonNull(aRCoreSceneformActivity2);
                        if (i == 0) {
                            Log.d("ARCoreSceneformActivity", "bitmapReady");
                            Date date = new Date();
                            DateFormat.format("yyyy-MM-dd_hh:mm:ss", date);
                            try {
                                str2 = aRCoreSceneformActivity2.getCacheDir().getAbsolutePath() + "/" + date + ".jpg";
                                FileOutputStream fileOutputStream = new FileOutputStream(new File(str2));
                                bitmap.compress(Bitmap.CompressFormat.JPEG, 80, fileOutputStream);
                                fileOutputStream.flush();
                                fileOutputStream.close();
                            } catch (Throwable th) {
                                th.printStackTrace();
                                str2 = null;
                            }
                            bitmap.recycle();
                            if (str2 != null) {
                                aRCoreSceneformActivity2.y(str2, false);
                                return;
                            }
                            return;
                        }
                        Log.e("ARCoreSceneformActivity", "captureImage error");
                    }
                }, new Handler());
            }
        });
        this.S.setOnLongClickListener(new View.OnLongClickListener() { // from class: c.e.b.q
            @Override // android.view.View.OnLongClickListener
            public final boolean onLongClick(View view) {
                ARCoreSceneformActivity aRCoreSceneformActivity = ARCoreSceneformActivity.this;
                if (aRCoreSceneformActivity.h0) {
                    Toast.makeText(aRCoreSceneformActivity.D, "Recording OFF", 1).show();
                    aRCoreSceneformActivity.b0.setText(aRCoreSceneformActivity.getResources().getString(R.string.photo_video));
                } else {
                    Toast.makeText(aRCoreSceneformActivity.D, "Recording ON", 1).show();
                    aRCoreSceneformActivity.b0.setText(aRCoreSceneformActivity.getResources().getString(R.string.stop_recording));
                }
                aRCoreSceneformActivity.v();
                return true;
            }
        });
        this.a0 = (TextView) findViewById(R.id.videoTimerTxt);
        Bundle extras = getIntent().getExtras();
        if (extras != null) {
            if (extras.containsKey("response")) {
                this.x = extras.getString("response");
            }
            if (extras.containsKey("alphaid")) {
                this.K = extras.getString("alphaid");
            }
            if (extras.containsKey("menuItemJson")) {
                this.L = extras.getString("menuItemJson");
            }
            if (extras.containsKey("groundContentId")) {
                this.M = extras.getString("groundContentId");
            }
        }
        this.D = this;
        getSharedPreferences("Unity", 0).edit();
        ArFragment arFragment = (ArFragment) m().H(R.id.sceneform_fragment);
        this.s = arFragment;
        arFragment.getArSceneView().getScene().addOnUpdateListener(this);
        this.s.getArSceneView().getScene().addOnPeekTouchListener(this);
        this.t = this.s.getArSceneView();
        new ExternalTexture();
        ArSceneView arSceneView = this.t;
        Node node = new Node();
        this.u = node;
        node.setParent(arSceneView.getScene());
        c.b.a.a.a.C(0.5f, 0.5f, 0.5f, this.u);
        this.u.setEnabled(false);
        Material.builder().setSource(this, R.raw.sceneform_opaque_colored_material).build().thenAccept((Consumer<? super Material>) c.e.b.d.f4619a).exceptionally(new Function() { // from class: c.e.b.k
            @Override // java.util.function.Function
            public final Object apply(Object obj) {
                Throwable th = (Throwable) obj;
                Objects.requireNonNull(ARCoreSceneformActivity.this);
                Log.e("ARCoreSceneformActivity", "Unable to load camera renderable");
                return null;
            }
        });
        ArSceneView arSceneView2 = this.t;
        SimpleTransformableNode simpleTransformableNode = new SimpleTransformableNode(this.s.getTransformationSystem());
        this.v = simpleTransformableNode;
        simpleTransformableNode.setParent(arSceneView2.getScene());
        this.v.setWorldPosition(new Vector3());
        this.v.setWorldRotation(new Quaternion());
        this.v.setLocalScale(new Vector3(1.0f, 1.0f, 1.0f));
        this.v.getScaleController().setMinScale(0.1f);
        this.v.getScaleController().setMaxScale(3.5f);
        this.v.setEnabled(false);
        ViewRenderable.builder().setView(this.D, R.layout.tap_imageview).build().thenAccept(new Consumer() { // from class: c.e.b.i
            @Override // java.util.function.Consumer
            public final void accept(Object obj) {
                ARCoreSceneformActivity aRCoreSceneformActivity = ARCoreSceneformActivity.this;
                ViewRenderable viewRenderable = (ViewRenderable) obj;
                Objects.requireNonNull(aRCoreSceneformActivity);
                c.c.a.b.b(aRCoreSceneformActivity).i.d(aRCoreSceneformActivity).j(Integer.valueOf((int) R.drawable.tap_animation)).B((ImageView) viewRenderable.getView().findViewById(R.id.img_loader_view));
                viewRenderable.setHorizontalAlignment(ViewRenderable.HorizontalAlignment.CENTER);
                c.b.a.a.a.J(viewRenderable, ViewRenderable.VerticalAlignment.CENTER, false, false);
                aRCoreSceneformActivity.v.setRenderable(viewRenderable);
            }
        }).exceptionally(new Function() { // from class: c.e.b.n
            @Override // java.util.function.Function
            public final Object apply(Object obj) {
                Throwable th = (Throwable) obj;
                Objects.requireNonNull(ARCoreSceneformActivity.this);
                Log.e("ARCoreSceneformActivity", "Unable to load  renderable");
                return null;
            }
        });
        this.v.setOnAccurateTapListner(new Node.OnAccurateTapListner() { // from class: c.e.b.v
            @Override // com.google.ar.sceneform.Node.OnAccurateTapListner
            public final void onAccurateTap(HitTestResult hitTestResult, MotionEvent motionEvent) {
                final ARCoreSceneformActivity aRCoreSceneformActivity = ARCoreSceneformActivity.this;
                AnchorNode anchorNode = (AnchorNode) aRCoreSceneformActivity.v.getParent();
                aRCoreSceneformActivity.u.setLocalPosition(aRCoreSceneformActivity.v.getLocalPosition());
                aRCoreSceneformActivity.u.setLocalRotation(aRCoreSceneformActivity.v.getLocalRotation());
                aRCoreSceneformActivity.u.setLocalScale(aRCoreSceneformActivity.v.getLocalScale().scaled(1.0f));
                anchorNode.removeChild(aRCoreSceneformActivity.v);
                aRCoreSceneformActivity.u.setParent(anchorNode);
                aRCoreSceneformActivity.u.setEnabled(true);
                aRCoreSceneformActivity.runOnUiThread(new Runnable() { // from class: c.e.b.j
                    @Override // java.lang.Runnable
                    public final void run() {
                        ARCoreSceneformActivity aRCoreSceneformActivity2 = ARCoreSceneformActivity.this;
                        aRCoreSceneformActivity2.A(false, true);
                        TranslateAnimation translateAnimation = new TranslateAnimation(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, -50.0f);
                        translateAnimation.setDuration(1000L);
                        translateAnimation.setFillAfter(false);
                        translateAnimation.setRepeatCount(2);
                        aRCoreSceneformActivity2.W.startAnimation(translateAnimation);
                    }
                });
                aRCoreSceneformActivity.runOnUiThread(new Runnable() { // from class: c.e.b.r
                    @Override // java.lang.Runnable
                    public final void run() {
                        ARCoreSceneformActivity aRCoreSceneformActivity2 = ARCoreSceneformActivity.this;
                        if (!aRCoreSceneformActivity2.P) {
                            aRCoreSceneformActivity2.P = true;
                            if (!aRCoreSceneformActivity2.K.isEmpty()) {
                                final yd ydVar = aRCoreSceneformActivity2.N;
                                ydVar.h();
                                try {
                                    JSONObject jSONObject = new JSONObject(ydVar.B);
                                    if (jSONObject.getBoolean(SettingsJsonConstants.APP_STATUS_KEY)) {
                                        JSONArray jSONArray = jSONObject.getJSONObject("data").getJSONArray("arContent");
                                        int length = jSONArray.length();
                                        ydVar.s = length;
                                        if (length == 0) {
                                            ydVar.f5451b.runOnUiThread(new Runnable() { // from class: c.e.b.s7
                                                @Override // java.lang.Runnable
                                                public final void run() {
                                                    yd.this.j();
                                                }
                                            });
                                        }
                                        for (int i = 0; i < jSONArray.length(); i++) {
                                            ydVar.g(jSONArray.getJSONObject(i));
                                        }
                                    } else {
                                        jSONObject.getString(InstallActivity.MESSAGE_TYPE_KEY);
                                        ydVar.f5451b.runOnUiThread(new Runnable() { // from class: c.e.b.ha
                                            @Override // java.lang.Runnable
                                            public final void run() {
                                                yd.this.j();
                                            }
                                        });
                                    }
                                } catch (JSONException e2) {
                                    e2.printStackTrace();
                                }
                                ydVar.r = System.currentTimeMillis();
                            }
                            if (aRCoreSceneformActivity2.M.isEmpty()) {
                                return;
                            }
                            final vc vcVar = aRCoreSceneformActivity2.O;
                            g gVar = new g(aRCoreSceneformActivity2);
                            Objects.requireNonNull(vcVar);
                            Node node2 = new Node();
                            vcVar.j = node2;
                            node2.setParent(vcVar.i);
                            vcVar.j.setLocalPosition(new Vector3(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 0.02f));
                            ViewRenderable.builder().setView(vcVar.f5340h, R.layout.ground_plane_loader).build().thenAccept(new Consumer() { // from class: c.e.b.z3
                                @Override // java.util.function.Consumer
                                public final void accept(Object obj) {
                                    vc vcVar2 = vc.this;
                                    ViewRenderable viewRenderable = (ViewRenderable) obj;
                                    Objects.requireNonNull(vcVar2);
                                    ((ProgressBar) viewRenderable.getView().findViewById(R.id.imageTargetLoaderProgressbar)).setIndeterminate(true);
                                    ((TextView) viewRenderable.getView().findViewById(R.id.imageTargetLoaderProgressbarText)).setText("");
                                    viewRenderable.setShadowCaster(false);
                                    viewRenderable.setShadowReceiver(false);
                                    viewRenderable.setHorizontalAlignment(ViewRenderable.HorizontalAlignment.CENTER);
                                    viewRenderable.setVerticalAlignment(ViewRenderable.VerticalAlignment.CENTER);
                                    vcVar2.j.setRenderable(viewRenderable);
                                    c.b.a.a.a.C(0.3f, 0.3f, 0.3f, vcVar2.j);
                                }
                            }).exceptionally(new Function() { // from class: c.e.b.y3
                                @Override // java.util.function.Function
                                public final Object apply(Object obj) {
                                    Throwable th = (Throwable) obj;
                                    Objects.requireNonNull(vc.this);
                                    Log.e("LoaderARContentGroundPlaneSceneformARCore", "Unable to load  renderable");
                                    return null;
                                }
                            });
                            String string = Settings.Secure.getString(vcVar.f5339g.getContentResolver(), "android_id");
                            vcVar.f5338f = new ec();
                            String v = c.b.a.a.a.v(new StringBuilder(), vcVar.f5337e, "arcontent/get_ar_bundle_app_hash");
                            u.a aVar = new u.a();
                            aVar.c(f.u.f6108b);
                            aVar.a("campaign_hash", vcVar.f5336d);
                            aVar.a("scan_mode", "1");
                            aVar.a("device_id", string);
                            f.u b2 = aVar.b();
                            StringBuilder A = c.b.a.a.a.A(v, "  ");
                            A.append(b2.toString());
                            Log.d("LoaderARContentGroundPlaneSceneformARCore", A.toString());
                            vcVar.f5338f.a(v, b2, new zc(vcVar, gVar));
                            System.currentTimeMillis();
                            return;
                        }
                        aRCoreSceneformActivity2.N.J.forEach(new ba("reload"));
                    }
                });
            }
        });
        this.w = new p();
        this.t.getScene().getSunlight();
        this.t.getPlaneRenderer().setVisible(false);
        if (!this.M.isEmpty()) {
            this.O = new vc(this.M, ac.f4547a.f4549c, this.u, this.s, this.D, this);
        } else {
            yd ydVar = new yd(this.K, this.x, ac.f4547a.f4549c, this.u, this.s, this.D, this);
            this.N = ydVar;
            ydVar.u = false;
        }
        String str2 = this.L;
        if (str2 != null) {
            if (!str2.isEmpty()) {
                StringBuilder x = c.b.a.a.a.x("menuItemJson - ");
                x.append(this.L);
                Log.d("ARCoreSceneformActivity", x.toString());
                String str3 = this.L;
                z();
                ArrayList<c.e.b.hf.a> arrayList = this.p0;
                if (arrayList != null) {
                    Iterator<c.e.b.hf.a> it = arrayList.iterator();
                    while (it.hasNext()) {
                        c.e.b.hf.a next = it.next();
                        if (next.file_loc.equals(str3)) {
                            str = next.id;
                            break;
                        }
                    }
                }
                str = "";
                File file = new File(getCacheDir(), "assets/models");
                if (!file.exists()) {
                    file.mkdirs();
                }
                this.d0.setVisibility(8);
                this.R.setVisibility(8);
                this.Z.setVisibility(8);
                this.c0.setVisibility(8);
                A(false, false);
                C("", false, 0);
                this.x0 = new Dialog(this, 16973838);
                View inflate = LayoutInflater.from(this).inflate(R.layout.download_progress, (ViewGroup) null);
                this.v0 = (ProgressBar) inflate.findViewById(R.id.progressBar2);
                this.w0 = (TextView) inflate.findViewById(R.id.downloadSize);
                this.v0.setProgress(0);
                this.x0.setContentView(inflate);
                this.x0.getWindow().setBackgroundDrawable(new ColorDrawable(0));
                File file2 = new File(getCacheDir(), c.b.a.a.a.r("assets/models/", str, ".glb"));
                StringBuilder x2 = c.b.a.a.a.x("file path --  ");
                x2.append(file2.getPath());
                Log.d("ARCoreSceneformActivity", x2.toString());
                if (file2.exists()) {
                    x(file2.getPath());
                    z = true;
                } else {
                    new j(this, str, true).execute(str3);
                    z = false;
                }
                this.x0.setOnDismissListener(new sb(this));
                if (!z) {
                    this.x0.show();
                } else {
                    A(true, false);
                    this.d0.setVisibility(0);
                }
                this.Z.setVisibility(8);
                this.c0.setVisibility(8);
            }
        } else {
            this.c0.setVisibility(8);
            A(true, false);
        }
        Matrix.setIdentityM(this.z, 0);
        SensorManager sensorManager = (SensorManager) getSystemService("sensor");
        r = sensorManager;
        Sensor defaultSensor = sensorManager.getDefaultSensor(4);
        this.G = defaultSensor;
        if (defaultSensor != null) {
            r.registerListener(this.z0, defaultSensor, 1);
        } else {
            Log.e("ARCoreSceneformActivity", "Registerered for ORIENTATION Sensor");
            Toast.makeText(this, "ORIENTATION Sensor not found", 1).show();
            finish();
        }
        DisplayMetrics displayMetrics = new DisplayMetrics();
        getWindowManager().getDefaultDisplay().getMetrics(displayMetrics);
        this.I = displayMetrics.heightPixels / 2;
        this.H = displayMetrics.widthPixels / 2;
        MediaPlayer.create(this, (int) R.raw.audio_screenshot);
        if (this.p0 == null && getIntent().hasExtra("fromargallery")) {
            z();
        }
        try {
            new Config(new Session(this)).setPlaneFindingMode(Config.PlaneFindingMode.HORIZONTAL);
        } catch (UnavailableApkTooOldException e2) {
            e2.printStackTrace();
        } catch (UnavailableArcoreNotInstalledException e3) {
            e3.printStackTrace();
        } catch (UnavailableDeviceNotCompatibleException e4) {
            e4.printStackTrace();
        } catch (UnavailableSdkTooOldException e5) {
            e5.printStackTrace();
        }
    }

    @Override // b.b.c.h, b.q.b.d, android.app.Activity
    public void onDestroy() {
        yd ydVar = this.N;
        if (ydVar != null) {
            ydVar.x = true;
            if (ydVar.y.length() != 0 && ydVar.u) {
                long currentTimeMillis = (System.currentTimeMillis() - ydVar.r) / 1000;
                ec ecVar = new ec();
                String v = c.b.a.a.a.v(new StringBuilder(), ydVar.z, "unitear_app/save_scan_spend_time");
                String string = Settings.Secure.getString(ydVar.f5450a.getContentResolver(), "android_id");
                u.a aVar = new u.a();
                aVar.c(u.f6108b);
                aVar.a("campaign_category_id", "1");
                aVar.a("unique_id", ydVar.y);
                aVar.a("time_spend", "" + currentTimeMillis);
                aVar.a("scan_mode", "1");
                aVar.a("device_id", string);
                u b2 = aVar.b();
                StringBuilder A = c.b.a.a.a.A(v, "  ");
                A.append(ydVar.y);
                A.append(" time = ");
                A.append(currentTimeMillis);
                Log.d("LoaderARContentSceneformARCore", A.toString());
                ecVar.a(v, b2, new rd(ydVar));
            }
        }
        super.onDestroy();
        if (this.G != null) {
            r.unregisterListener(this.z0);
        }
        Log.e("ARCoreSceneformActivity", "onDestroy");
        Log.e("ARCoreSceneformActivity", "memory free = " + Runtime.getRuntime().freeMemory());
    }

    @Override // b.q.b.d, android.app.Activity, android.content.ComponentCallbacks
    public void onLowMemory() {
        super.onLowMemory();
        c.e.b.p000if.f.a(this.D, 60);
        Log.d("ARCoreSceneformActivity", "Clear cached 3D models");
    }

    @Override // b.q.b.d, android.app.Activity
    public void onPause() {
        if (this.h0) {
            this.F = false;
            this.l0.f4904c.pause();
        }
        yd ydVar = this.N;
        if (ydVar != null) {
            ydVar.q();
        }
        super.onPause();
    }

    @Override // com.google.ar.sceneform.Scene.OnPeekTouchListener
    public void onPeekTouch(HitTestResult hitTestResult, MotionEvent motionEvent) {
        int actionMasked = motionEvent.getActionMasked();
        if (actionMasked == 0) {
            this.A = motionEvent.getX();
            this.B = motionEvent.getY();
        } else if (actionMasked == 1) {
            float x = motionEvent.getX();
            if (this.B > motionEvent.getY() && this.c0.getVisibility() == 0) {
                int[] iArr = new int[2];
                this.c0.getLocationOnScreen(iArr);
                int i = iArr[0];
                Log.d("test36", this.A + "  " + x);
                float abs = Math.abs(this.A - x);
                if (this.B >= iArr[1] - 100 && abs < 220.0f) {
                    B();
                }
            }
        }
        for (Node node : this.u.getChildren()) {
            if (node instanceof SimpleTransformableNode) {
                SimpleTransformableNode simpleTransformableNode = (SimpleTransformableNode) node;
                if (simpleTransformableNode.isSelected()) {
                    this.w.a(motionEvent, simpleTransformableNode);
                }
            }
        }
    }

    @Override // b.q.b.d, android.app.Activity
    public void onRequestPermissionsResult(int i, String[] strArr, int[] iArr) {
        super.onRequestPermissionsResult(i, strArr, iArr);
        if (b.j.c.a.a(this, "android.permission.CAMERA") == 0) {
            return;
        }
        Toast.makeText(this, "Camera permission is needed to run this application", 1).show();
        int i2 = b.j.b.a.f2030b;
        if (!shouldShowRequestPermissionRationale("android.permission.CAMERA")) {
            Intent intent = new Intent();
            intent.setAction("android.settings.APPLICATION_DETAILS_SETTINGS");
            intent.setData(Uri.fromParts("package", getPackageName(), null));
            startActivity(intent);
        }
        finish();
    }

    @Override // b.q.b.d, android.app.Activity
    public void onResume() {
        super.onResume();
        getWindow().getDecorView().setSystemUiVisibility(5894);
        ArFragment arFragment = this.s;
        if (arFragment != null) {
            arFragment.getPlaneDiscoveryController().hide();
            this.s.getPlaneDiscoveryController().setInstructionView(null);
        }
        yd ydVar = this.N;
        if (ydVar != null) {
            ydVar.J.forEach(new ba("resume"));
        }
        if (this.h0) {
            this.l0.f4904c.resume();
            this.F = true;
        }
    }

    @Override // b.b.c.h, b.q.b.d, android.app.Activity
    public void onStop() {
        super.onStop();
    }

    @Override // com.google.ar.sceneform.Scene.OnUpdateListener
    public void onUpdate(FrameTime frameTime) {
        Frame arFrame = this.t.getArFrame();
        Camera camera = arFrame.getCamera();
        if (camera.getTrackingState() != TrackingState.TRACKING || this.C) {
            return;
        }
        StringBuilder x = c.b.a.a.a.x("Searching Anchor ");
        x.append(this.H);
        x.append(", ");
        x.append(this.I);
        x.append(" Tracking state = ");
        x.append(camera.getTrackingState());
        Log.d("ARCoreSceneformActivity", x.toString());
        for (HitResult hitResult : arFrame.hitTest(this.H, this.I)) {
            Trackable trackable = hitResult.getTrackable();
            if (trackable instanceof Plane) {
                Plane plane = (Plane) trackable;
                if (plane.isPoseInPolygon(hitResult.getHitPose())) {
                    int i = this.J;
                    if (i < 15) {
                        this.J = i + 1;
                        return;
                    }
                    if (this.y.size() >= 1) {
                        if (plane.getType() == Plane.Type.VERTICAL) {
                            this.y0 = "Vertical";
                        } else {
                            this.y0 = "Horizontal";
                        }
                    }
                    this.C = true;
                    Anchor createAnchor = hitResult.createAnchor();
                    this.y.add(createAnchor);
                    AnchorNode anchorNode = new AnchorNode(createAnchor);
                    anchorNode.setParent(this.t.getScene());
                    this.v.setParent(anchorNode);
                    Quaternion axisAngle = Quaternion.axisAngle(this.v.getRight(), 180.0f);
                    SimpleTransformableNode simpleTransformableNode = this.v;
                    simpleTransformableNode.setLocalRotation(Quaternion.multiply(axisAngle, simpleTransformableNode.getLocalRotation()));
                    if (this.y0.equals("Vertical")) {
                        this.v.setLookDirection(Vector3.zero(), anchorNode.getDown());
                    } else {
                        this.v.setLookDirection(Vector3.down(), anchorNode.getUp());
                    }
                    this.v.setEnabled(true);
                    runOnUiThread(new Runnable() { // from class: c.e.b.l
                        @Override // java.lang.Runnable
                        public final void run() {
                            ARCoreSceneformActivity aRCoreSceneformActivity = ARCoreSceneformActivity.this;
                            aRCoreSceneformActivity.X.setVisibility(8);
                            aRCoreSceneformActivity.V.clearAnimation();
                            aRCoreSceneformActivity.d0.setVisibility(8);
                            aRCoreSceneformActivity.C(aRCoreSceneformActivity.getResources().getString(R.string.surface_found), true, 2000);
                        }
                    });
                    Log.d("ARCoreSceneformActivity", "Anchor Added " + this.y.size());
                    return;
                }
            }
        }
    }

    @Override // android.app.Activity, android.view.Window.Callback
    public void onWindowFocusChanged(boolean z) {
        super.onWindowFocusChanged(z);
        if (z) {
            getWindow().getDecorView().setSystemUiVisibility(5894);
        }
    }

    public final void v() {
        new Handler().postDelayed(new Runnable() { // from class: c.e.b.s
            @Override // java.lang.Runnable
            public final void run() {
                ARCoreSceneformActivity aRCoreSceneformActivity = ARCoreSceneformActivity.this;
                aRCoreSceneformActivity.a0.setVisibility(0);
                aRCoreSceneformActivity.g0 = 0;
                aRCoreSceneformActivity.f0 = 0;
                aRCoreSceneformActivity.a0.setText(String.format("%02d:%02d", 0, Integer.valueOf(aRCoreSceneformActivity.f0)));
                if (aRCoreSceneformActivity.l0 == null) {
                    c.e.b.p000if.q qVar = new c.e.b.p000if.q(aRCoreSceneformActivity.D);
                    aRCoreSceneformActivity.l0 = qVar;
                    qVar.f4906e = aRCoreSceneformActivity.s.getArSceneView();
                    int i = aRCoreSceneformActivity.getResources().getConfiguration().orientation;
                    if (aRCoreSceneformActivity.H * 2 > 1000) {
                        aRCoreSceneformActivity.l0.d(6, i);
                    } else {
                        aRCoreSceneformActivity.l0.d(5, i);
                    }
                }
                if (!aRCoreSceneformActivity.h0) {
                    boolean b2 = aRCoreSceneformActivity.l0.b();
                    aRCoreSceneformActivity.h0 = b2;
                    if (b2) {
                        aRCoreSceneformActivity.F = true;
                        Timer timer = aRCoreSceneformActivity.j0;
                        if (timer != null) {
                            timer.cancel();
                        }
                        aRCoreSceneformActivity.g0 = 0;
                        aRCoreSceneformActivity.f0 = -1;
                        Timer timer2 = new Timer();
                        aRCoreSceneformActivity.j0 = timer2;
                        timer2.scheduleAtFixedRate(new rb(aRCoreSceneformActivity), 0L, 1000L);
                        aRCoreSceneformActivity.w(true);
                        return;
                    }
                    return;
                }
                aRCoreSceneformActivity.h0 = aRCoreSceneformActivity.l0.b();
                aRCoreSceneformActivity.w(false);
                aRCoreSceneformActivity.f0 = 0;
                aRCoreSceneformActivity.g0 = 0;
                Timer timer3 = aRCoreSceneformActivity.j0;
                if (timer3 != null) {
                    timer3.cancel();
                }
                String path = aRCoreSceneformActivity.l0.i.getPath();
                aRCoreSceneformActivity.i0 = path;
                aRCoreSceneformActivity.y(path, true);
            }
        }, 300L);
    }

    public final void w(boolean z) {
        if (!z) {
            this.a0.setVisibility(8);
        }
        if (z) {
            ImageButton imageButton = this.S;
            Object obj = b.j.c.a.f2074a;
            imageButton.setImageDrawable(getDrawable(R.drawable.ic_video_recording));
            this.c0.setVisibility(8);
            this.k0 = new l(this, this.t);
            return;
        }
        ImageButton imageButton2 = this.S;
        Object obj2 = b.j.c.a.f2074a;
        imageButton2.setImageDrawable(getDrawable(R.drawable.camerabutton));
        if (this.L != null) {
            this.c0.setVisibility(0);
        }
        l lVar = this.k0;
        if (lVar != null) {
            lVar.a();
        }
    }

    public final void x(String str) {
        final yd ydVar = this.N;
        float[] fArr = {StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD};
        float[] fArr2 = {StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 0.707f, -0.707f, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD};
        ydVar.x = false;
        Node node = new Node();
        SimpleTransformableNode simpleTransformableNode = new SimpleTransformableNode(ydVar.f5456g.getTransformationSystem());
        simpleTransformableNode.setParent(ydVar.f5453d);
        simpleTransformableNode.setLocalPosition(new Vector3(fArr[0], fArr[1], fArr[2]));
        simpleTransformableNode.setLocalRotation(Quaternion.eulerAngles(new Vector3(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD)));
        node.setParent(simpleTransformableNode);
        node.setLocalRotation(Quaternion.multiply(Quaternion.eulerAngles(new Vector3(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 180.0f)), new Quaternion(fArr2[0], fArr2[1], fArr2[2], fArr2[3])));
        final Node node2 = new Node();
        node2.setParent(node);
        node2.setLocalRotation(Quaternion.eulerAngles(new Vector3(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 180.0f, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD)));
        ydVar.f5456g.getArSceneView().getScene().addOnPeekTouchListener(new Scene.OnPeekTouchListener() { // from class: c.e.b.u8
            @Override // com.google.ar.sceneform.Scene.OnPeekTouchListener
            public final void onPeekTouch(HitTestResult hitTestResult, MotionEvent motionEvent) {
                yd.this.f5456g.getTransformationSystem().onTouch(hitTestResult, motionEvent);
            }
        });
        simpleTransformableNode.setLocalScale(new Vector3(0.5f, 0.5f, 0.5f));
        simpleTransformableNode.getScaleController().setMinScale(0.12f);
        simpleTransformableNode.getScaleController().setMaxScale(2.5f);
        Log.d("LoaderARContentSceneformARCore", "load3Dmodel model uri " + Uri.fromFile(new File(str)));
        ModelRenderable.builder().setSource(ydVar.f5451b, Uri.fromFile(new File(str))).setIsFilamentGltf(true).build().thenAccept(new Consumer() { // from class: c.e.b.l8
            @Override // java.util.function.Consumer
            public final void accept(Object obj) {
                yd ydVar2 = yd.this;
                Node node3 = node2;
                Objects.requireNonNull(ydVar2);
                Log.d("LoaderARContentSceneformARCore", "load3Dmodel model loaded");
                node3.setRenderable((ModelRenderable) obj);
                ydVar2.e(node3.getRenderableInstance(), "REPEAT", 1, "ALL", true, "model");
                ydVar2.u(node3, null);
            }
        }).exceptionally(new Function() { // from class: c.e.b.b8
            @Override // java.util.function.Function
            public final Object apply(Object obj) {
                Objects.requireNonNull(yd.this);
                StringBuilder sb = new StringBuilder();
                sb.append("load3Dmodel ");
                c.b.a.a.a.N((Throwable) obj, sb, "LoaderARContentSceneformARCore");
                return null;
            }
        });
    }

    public void y(String str, boolean z) {
        Intent intent = new Intent(this.D, CapturePreview.class);
        if (z) {
            intent.putExtra("videoUrl", str);
        } else {
            intent.putExtra("imageUrl", str);
        }
        startActivity(intent);
    }

    public void z() {
        String str = ARGalleryActivity.s;
        if (str != null) {
            this.p0 = new ArrayList<>();
            this.p0 = c.e.b.hf.a.a(str);
            this.q0 = new ArrayList<>();
            Iterator<c.e.b.hf.a> it = this.p0.iterator();
            while (it.hasNext()) {
                c.e.b.hf.a next = it.next();
                boolean z = true;
                if (this.u0.contains(next.file_loc)) {
                    next.isLoaded = true;
                }
                Iterator<d> it2 = this.q0.iterator();
                while (true) {
                    if (!it2.hasNext()) {
                        z = false;
                        break;
                    } else if (next.category.equals(it2.next().name)) {
                        break;
                    }
                }
                if (!z) {
                    this.q0.add(new d(next.category, false));
                }
            }
        }
    }
}