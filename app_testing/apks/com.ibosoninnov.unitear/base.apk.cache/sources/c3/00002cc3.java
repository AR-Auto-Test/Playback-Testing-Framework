package com.ibosoninnov.unitear;

import android.app.Activity;
import android.app.ActivityManager;
import android.app.Dialog;
import android.app.NotificationManager;
import android.content.Context;
import android.content.Intent;
import android.graphics.drawable.ColorDrawable;
import android.media.MediaPlayer;
import android.net.Uri;
import android.opengl.Matrix;
import android.os.Bundle;
import android.os.Handler;
import android.provider.Settings;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.RelativeLayout;
import android.widget.TextView;
import android.widget.Toast;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.recyclerview.widget.RecyclerView;
import b.b.c.h;
import c.e.b.ef.f;
import c.e.b.ge;
import c.e.b.hd;
import c.e.b.ie;
import c.e.b.jc;
import c.e.b.je;
import c.e.b.ke;
import c.e.b.me;
import c.e.b.ne;
import c.e.b.oe;
import c.e.b.p000if.d;
import c.e.b.p000if.j;
import c.e.b.p000if.l;
import c.e.b.p000if.p;
import c.e.b.p000if.q;
import c.e.b.pe;
import c.e.b.qe;
import c.e.b.re;
import com.google.android.material.bottomsheet.BottomSheetDialog;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.google.ar.sceneform.FrameTime;
import com.google.ar.sceneform.HitTestResult;
import com.google.ar.sceneform.Node;
import com.google.ar.sceneform.Scene;
import com.google.ar.sceneform.SimpleSceneView;
import com.google.ar.sceneform.math.Quaternion;
import com.google.ar.sceneform.math.Vector3;
import com.google.ar.sceneform.rendering.ExternalTexture;
import com.google.ar.sceneform.rendering.Material;
import com.google.ar.sceneform.rendering.ModelRenderable;
import com.google.ar.sceneform.rendering.ShapeFactory;
import com.google.ar.sceneform.rendering.ViewRenderable;
import com.google.ar.sceneform.ux.SimpleFragment;
import com.google.ar.sceneform.ux.SimpleTransformableNode;
import com.ibosoninnov.instanttrackinglib.InstantTrackingHelper;
import com.ibosoninnov.unitear.NonARCoreActivitySceneform;
import com.ibosoninnov.unitear.R;
import f.u;
import java.io.File;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Objects;
import java.util.Timer;
import java.util.function.Consumer;
import java.util.function.Function;

/* loaded from: classes2.dex */
public class NonARCoreActivitySceneform extends h implements c.e.b.gf.a, c.e.b.gf.b, Scene.OnPeekTouchListener, Scene.OnUpdateListener {
    public static final /* synthetic */ int r = 0;
    public SimpleFragment A;
    public SimpleSceneView B;
    public Node C;
    public Node D;
    public p F;
    public hd J;
    public jc K;
    public ImageButton L;
    public ImageButton M;
    public ImageButton N;
    public ImageButton O;
    public ImageButton P;
    public RelativeLayout Q;
    public LinearLayout R;
    public TextView S;
    public TextView T;
    public ConstraintLayout U;
    public boolean X;
    public String Y;
    public Timer Z;
    public l a0;
    public q b0;
    public d c0;
    public BottomSheetDialog d0;
    public RecyclerView e0;
    public RecyclerView f0;
    public c.e.b.ef.c i0;
    public f j0;
    public boolean m0;
    public ProgressBar n0;
    public TextView o0;
    public Dialog p0;
    public InstantTrackingHelper q0;
    public float r0;
    public float s0;
    public float t;
    public float u;
    public ProgressBar v;
    public long v0;
    public ImageView w;
    public Bundle w0;
    public Context x;
    public Activity y;
    public ExternalTexture z;
    public final float[] s = new float[16];
    public Quaternion E = Quaternion.eulerAngles(new Vector3(180.0f, 90.0f, 90.0f));
    public String G = "";
    public String H = null;
    public String I = "";
    public int V = 0;
    public int W = 0;
    public ArrayList<c.e.b.hf.a> g0 = null;
    public ArrayList<c.e.b.hf.d> h0 = null;
    public ArrayList<c.e.b.hf.a> k0 = new ArrayList<>();
    public ArrayList<String> l0 = new ArrayList<>();
    public boolean t0 = false;
    public boolean u0 = false;

    /* loaded from: classes2.dex */
    public class a implements Runnable {
        public a() {
        }

        @Override // java.lang.Runnable
        public void run() {
            NonARCoreActivitySceneform.this.Q.setVisibility(8);
            NonARCoreActivitySceneform.this.v.setVisibility(8);
            NonARCoreActivitySceneform.this.w.setVisibility(8);
        }
    }

    /* loaded from: classes2.dex */
    public class b implements InstantTrackingHelper.TrackingListener {
        public b() {
        }

        @Override // com.ibosoninnov.instanttrackinglib.InstantTrackingHelper.TrackingListener
        public void onTracking(List<Float> list) {
            if (list != null) {
                NonARCoreActivitySceneform nonARCoreActivitySceneform = NonARCoreActivitySceneform.this;
                int i = NonARCoreActivitySceneform.r;
                Objects.requireNonNull(nonARCoreActivitySceneform);
                nonARCoreActivitySceneform.D.setWorldPosition(new Vector3(list.get(3).floatValue() * 0.05f, (-list.get(7).floatValue()) * 0.05f, list.get(11).floatValue() * 0.05f));
                Vector3 vector3 = new Vector3(list.get(0).floatValue(), -list.get(4).floatValue(), list.get(8).floatValue());
                new Vector3(-list.get(1).floatValue(), list.get(5).floatValue(), list.get(9).floatValue());
                Vector3 vector32 = new Vector3(-list.get(2).floatValue(), list.get(6).floatValue(), -list.get(10).floatValue());
                if (!nonARCoreActivitySceneform.u0) {
                    nonARCoreActivitySceneform.u0 = true;
                    float atan2 = ((float) ((Math.atan2(-list.get(4).floatValue(), list.get(0).floatValue()) * 180.0d) / 3.141592653589793d)) + 90.0f;
                    Log.d("NonARCoreActivity", "Angle = " + atan2);
                    nonARCoreActivitySceneform.E = Quaternion.eulerAngles(new Vector3(180.0f, 90.0f, 90.0f - atan2));
                }
                Quaternion lookRotation = Quaternion.lookRotation(vector3, vector32);
                Node node = nonARCoreActivitySceneform.D;
                node.setWorldRotation(Quaternion.slerp(node.getWorldRotation(), Quaternion.multiply(lookRotation, nonARCoreActivitySceneform.E), 0.5f));
                NonARCoreActivitySceneform nonARCoreActivitySceneform2 = NonARCoreActivitySceneform.this;
                if (nonARCoreActivitySceneform2.t0) {
                    return;
                }
                nonARCoreActivitySceneform2.t0 = true;
                nonARCoreActivitySceneform2.runOnUiThread(new Runnable() { // from class: c.e.b.ab
                    @Override // java.lang.Runnable
                    public final void run() {
                        NonARCoreActivitySceneform.b bVar = NonARCoreActivitySceneform.b.this;
                        NonARCoreActivitySceneform nonARCoreActivitySceneform3 = NonARCoreActivitySceneform.this;
                        int i2 = NonARCoreActivitySceneform.r;
                        nonARCoreActivitySceneform3.C("", false, 0);
                        NonARCoreActivitySceneform.this.D.setEnabled(true);
                    }
                });
            }
        }
    }

    /* loaded from: classes2.dex */
    public class c implements Runnable {
        public c() {
        }

        @Override // java.lang.Runnable
        public void run() {
            NonARCoreActivitySceneform.this.q0.resetAnchor(0.5f, 0.5f);
        }
    }

    public static void v(NonARCoreActivitySceneform nonARCoreActivitySceneform) {
        Objects.requireNonNull(nonARCoreActivitySceneform);
        new Handler().postDelayed(new ge(nonARCoreActivitySceneform), 300L);
    }

    public void A() {
        String str = ARGalleryActivity.s;
        if (str != null) {
            this.g0 = new ArrayList<>();
            this.g0 = c.e.b.hf.a.a(str);
            this.h0 = new ArrayList<>();
            Iterator<c.e.b.hf.a> it = this.g0.iterator();
            while (it.hasNext()) {
                c.e.b.hf.a next = it.next();
                boolean z = true;
                if (this.l0.contains(next.file_loc)) {
                    next.isLoaded = true;
                }
                Iterator<c.e.b.hf.d> it2 = this.h0.iterator();
                while (true) {
                    if (!it2.hasNext()) {
                        z = false;
                        break;
                    } else if (next.category.equals(it2.next().name)) {
                        break;
                    }
                }
                if (!z) {
                    this.h0.add(new c.e.b.hf.d(next.category, false));
                }
            }
        }
    }

    public void B(String str) {
        String str2;
        A();
        ArrayList<c.e.b.hf.a> arrayList = this.g0;
        if (arrayList != null) {
            Iterator<c.e.b.hf.a> it = arrayList.iterator();
            while (it.hasNext()) {
                c.e.b.hf.a next = it.next();
                if (next.file_loc.equals(str)) {
                    str2 = next.id;
                    break;
                }
            }
        }
        str2 = "";
        File file = new File(getCacheDir(), "assets/models");
        if (!file.exists()) {
            file.mkdirs();
        }
        this.N.setVisibility(8);
        this.M.setVisibility(8);
        this.R.setVisibility(8);
        this.U.setVisibility(8);
        boolean z = false;
        if (this.t0) {
            C("", false, 0);
        }
        this.p0 = new Dialog(this, 16973838);
        View inflate = LayoutInflater.from(this).inflate(R.layout.download_progress, (ViewGroup) null);
        this.n0 = (ProgressBar) inflate.findViewById(R.id.progressBar2);
        this.o0 = (TextView) inflate.findViewById(R.id.downloadSize);
        this.n0.setProgress(0);
        this.p0.setContentView(inflate);
        this.p0.getWindow().setBackgroundDrawable(new ColorDrawable(0));
        File file2 = new File(getCacheDir(), c.b.a.a.a.r("assets/models/", str2, ".glb"));
        StringBuilder x = c.b.a.a.a.x("file path --  ");
        x.append(file2.getPath());
        Log.d("NonARCoreActivity", x.toString());
        if (file2.exists()) {
            y(file2.getPath());
            this.N.setVisibility(0);
            this.M.setVisibility(0);
            this.U.setVisibility(0);
            this.R.setVisibility(0);
            z = true;
        } else {
            new j(this, str2, true).execute(str);
        }
        this.p0.setOnDismissListener(new ke(this));
        if (z) {
            return;
        }
        this.p0.show();
    }

    public final void C(String str, boolean z, int i) {
        Log.d("NonARCoreActivity", "showTopMessage - " + str);
        ((TextView) findViewById(R.id.msgTxt)).setText(str);
        if (z) {
            this.Q.setVisibility(0);
        } else {
            new Handler().postDelayed(new a(), i);
        }
    }

    @Override // c.e.b.gf.a
    public void c(c.e.b.hf.a aVar) {
        this.d0.dismiss();
        StringBuilder sb = new StringBuilder();
        sb.append(this.x.getCacheDir());
        sb.append("/assets/models/");
        String v = c.b.a.a.a.v(sb, aVar.id, ".glb");
        hd hdVar = this.J;
        hdVar.x = true;
        for (Node node : hdVar.i.getChildren()) {
            node.setParent(null);
        }
        y(v);
    }

    @Override // c.e.b.gf.a
    public void d(c.e.b.hf.a aVar) {
        File file = new File(getCacheDir(), "assets/models");
        if (!file.exists()) {
            file.mkdir();
        }
        if (new File(this.x.getCacheDir(), c.b.a.a.a.v(c.b.a.a.a.x("assets/models/"), aVar.id, ".glb")).exists()) {
            f(aVar.id, 101, false, "");
        } else {
            new j(this, aVar.id, false).execute(aVar.file_loc);
        }
    }

    @Override // c.e.b.gf.b
    public void f(String str, int i, boolean z, String str2) {
        try {
            if (z) {
                if (this.p0.isShowing()) {
                    this.n0.setProgress(i);
                    this.o0.setText(str2);
                    if (i == 101) {
                        this.p0.dismiss();
                        String str3 = this.x.getCacheDir().getPath() + "/assets/models/" + str + ".glb";
                        Log.d("NonARCoreActivity", "file path --  " + str3);
                        y(str3);
                    }
                }
            } else {
                for (int i2 = 0; i2 <= this.k0.size(); i2++) {
                    if (this.k0.get(i2).id.equals(str)) {
                        this.k0.get(i2).downloadStatus = i;
                        this.j0.notifyItemChanged(i2);
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
        for (int i = 0; i < this.h0.size(); i++) {
            this.h0.get(i).isSelected = this.h0.get(i).name.equals(str);
        }
        this.i0.notifyDataSetChanged();
        this.k0.clear();
        Iterator<c.e.b.hf.a> it = this.g0.iterator();
        while (it.hasNext()) {
            c.e.b.hf.a next = it.next();
            if (next.category.equals(str)) {
                this.k0.add(next);
            }
            if (new File(getCacheDir(), c.b.a.a.a.v(c.b.a.a.a.x("models/"), next.id, ".glb")).exists()) {
                next.downloadStatus = 101;
            }
        }
        f fVar = new f(this.k0, this);
        this.j0 = fVar;
        this.f0.setAdapter(fVar);
    }

    @Override // androidx.activity.ComponentActivity, android.app.Activity
    public void onBackPressed() {
        boolean z = this.X;
        if (z) {
            Toast.makeText(this.x, "Recording OFF", 1).show();
            new Handler().postDelayed(new ge(this), 300L);
            return;
        }
        if (!z) {
            this.D.setParent(null);
        }
        this.f41f.b();
    }

    @Override // b.b.c.h, b.q.b.d, androidx.activity.ComponentActivity, b.j.b.e, android.app.Activity
    public void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        requestWindowFeature(1);
        getWindow().setFlags(1024, 1024);
        getWindow().addFlags(128);
        getWindow().addFlags(1536);
        setContentView(R.layout.activity_nonarcore_sceneform);
        ConstraintLayout constraintLayout = (ConstraintLayout) findViewById(R.id.swipeLayout);
        this.U = constraintLayout;
        constraintLayout.setVisibility(8);
        this.N = (ImageButton) findViewById(R.id.no_ar_help);
        this.O = (ImageButton) findViewById(R.id.reloadbtn);
        this.R = (LinearLayout) findViewById(R.id.arobjectfoundlayout);
        RelativeLayout relativeLayout = (RelativeLayout) findViewById(R.id.scanTheFloorLayout);
        ImageView imageView = (ImageView) findViewById(R.id.scanFloorImg);
        this.Q = (RelativeLayout) findViewById(R.id.messageLayout);
        this.v = (ProgressBar) findViewById(R.id.pgrbar);
        this.w = (ImageView) findViewById(R.id.pgrimg);
        ImageButton imageButton = (ImageButton) findViewById(R.id.closeARContentButton);
        this.L = imageButton;
        imageButton.setOnClickListener(new me(this));
        this.N.setOnClickListener(new ne(this));
        this.O.setOnClickListener(new oe(this));
        ImageButton imageButton2 = (ImageButton) findViewById(R.id.groundplaneButton);
        this.M = imageButton2;
        imageButton2.setOnClickListener(new pe(this));
        this.P = (ImageButton) findViewById(R.id.photoVideoSwitch);
        this.T = (TextView) findViewById(R.id.takePhotoVideo);
        this.P.setOnClickListener(new qe(this));
        this.P.setOnLongClickListener(new re(this));
        this.S = (TextView) findViewById(R.id.videoTimerTxt);
        Bundle extras = getIntent().getExtras();
        this.w0 = extras;
        if (extras != null) {
            if (extras.containsKey("alphaid")) {
                this.G = this.w0.getString("alphaid");
            }
            if (this.w0.containsKey("menuItemJson")) {
                this.H = this.w0.getString("menuItemJson");
                StringBuilder x = c.b.a.a.a.x("menuItemJson - ");
                x.append(this.H);
                Log.d("NonARCoreActivity", x.toString());
            }
            if (this.w0.containsKey("groundContentId")) {
                this.I = this.w0.getString("groundContentId");
            }
        }
        this.x = this;
        this.y = this;
        d dVar = new d(this);
        this.c0 = dVar;
        Integer.valueOf(dVar.f4871a.getInt("sensorOrientation", 0)).intValue();
        DisplayMetrics displayMetrics = new DisplayMetrics();
        getWindowManager().getDefaultDisplay().getMetrics(displayMetrics);
        this.s0 = displayMetrics.heightPixels;
        this.r0 = displayMetrics.widthPixels;
        C("Loading", true, 0);
        SimpleFragment simpleFragment = (SimpleFragment) m().H(R.id.simple_sceneform_fragment);
        this.A = simpleFragment;
        simpleFragment.getArSceneView().getScene().addOnUpdateListener(this);
        this.A.getArSceneView().getScene().addOnPeekTouchListener(this);
        this.B = this.A.getArSceneView();
        Node node = new Node();
        this.C = node;
        node.setWorldPosition(new Vector3(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, -10.0f));
        this.C.setParent(this.B.getScene().getCamera());
        this.B.getScene().getCamera().setVerticalFovDegrees(60.0f);
        this.z = new ExternalTexture();
        w();
        SimpleSceneView simpleSceneView = this.B;
        Node node2 = new Node();
        this.D = node2;
        node2.setParent(simpleSceneView.getScene());
        this.D.setLocalScale(new Vector3(0.5f, 0.5f, 0.5f));
        ViewRenderable.builder().setView(this.x, R.layout.tap_imageview).build().thenAccept(new Consumer() { // from class: c.e.b.db
            @Override // java.util.function.Consumer
            public final void accept(Object obj) {
                NonARCoreActivitySceneform nonARCoreActivitySceneform = NonARCoreActivitySceneform.this;
                ViewRenderable viewRenderable = (ViewRenderable) obj;
                Objects.requireNonNull(nonARCoreActivitySceneform);
                c.c.a.b.b(nonARCoreActivitySceneform).i.d(nonARCoreActivitySceneform).j(Integer.valueOf((int) R.drawable.tap_animation)).B((ImageView) viewRenderable.getView().findViewById(R.id.img_loader_view));
                viewRenderable.setHorizontalAlignment(ViewRenderable.HorizontalAlignment.CENTER);
                c.b.a.a.a.J(viewRenderable, ViewRenderable.VerticalAlignment.CENTER, false, false);
                nonARCoreActivitySceneform.D.setRenderable(viewRenderable);
            }
        }).exceptionally(new Function() { // from class: c.e.b.za
            @Override // java.util.function.Function
            public final Object apply(Object obj) {
                Throwable th = (Throwable) obj;
                Objects.requireNonNull(NonARCoreActivitySceneform.this);
                Log.e("NonARCoreActivity", "Unable to load camera renderable");
                return null;
            }
        });
        this.D.setOnAccurateTapListner(new Node.OnAccurateTapListner() { // from class: c.e.b.gb
            @Override // com.google.ar.sceneform.Node.OnAccurateTapListner
            public final void onAccurateTap(HitTestResult hitTestResult, MotionEvent motionEvent) {
                final NonARCoreActivitySceneform nonARCoreActivitySceneform = NonARCoreActivitySceneform.this;
                nonARCoreActivitySceneform.D.setRenderable(null);
                nonARCoreActivitySceneform.R.setVisibility(0);
                c.b.a.a.a.C(3.0f, 3.0f, 3.0f, nonARCoreActivitySceneform.D);
                if (!nonARCoreActivitySceneform.I.isEmpty()) {
                    nonARCoreActivitySceneform.K = new jc(nonARCoreActivitySceneform.I, ac.f4547a.f4549c, nonARCoreActivitySceneform.D, nonARCoreActivitySceneform.A, nonARCoreActivitySceneform.x, nonARCoreActivitySceneform.y);
                } else {
                    hd hdVar = new hd(nonARCoreActivitySceneform.G, ac.f4547a.f4549c, nonARCoreActivitySceneform.D, nonARCoreActivitySceneform.A, nonARCoreActivitySceneform.x, nonARCoreActivitySceneform.y);
                    nonARCoreActivitySceneform.J = hdVar;
                    hdVar.u = false;
                }
                String str = nonARCoreActivitySceneform.H;
                if (str != null) {
                    if (str.isEmpty()) {
                        return;
                    }
                    Log.d("NonARCoreActivity", nonARCoreActivitySceneform.H);
                    String str2 = nonARCoreActivitySceneform.H;
                    nonARCoreActivitySceneform.w0.getString("id");
                    nonARCoreActivitySceneform.B(str2);
                    return;
                }
                nonARCoreActivitySceneform.U.setVisibility(8);
                if (!nonARCoreActivitySceneform.G.isEmpty()) {
                    nonARCoreActivitySceneform.J.u(new hd.g() { // from class: c.e.b.bb
                        @Override // c.e.b.hd.g
                        public final void a(String str3) {
                            final NonARCoreActivitySceneform nonARCoreActivitySceneform2 = NonARCoreActivitySceneform.this;
                            Objects.requireNonNull(nonARCoreActivitySceneform2);
                            Log.d("NonARCoreActivity", "loaderARContent - " + str3);
                            nonARCoreActivitySceneform2.q0.resetAnchor(0.5f, 0.5f);
                            nonARCoreActivitySceneform2.runOnUiThread(new Runnable() { // from class: c.e.b.ib
                                @Override // java.lang.Runnable
                                public final void run() {
                                    NonARCoreActivitySceneform.this.R.setVisibility(0);
                                }
                            });
                        }
                    });
                }
                if (!nonARCoreActivitySceneform.I.isEmpty()) {
                    final jc jcVar = nonARCoreActivitySceneform.K;
                    cb cbVar = new cb(nonARCoreActivitySceneform);
                    Objects.requireNonNull(jcVar);
                    Node node3 = new Node();
                    jcVar.j = node3;
                    node3.setParent(jcVar.i);
                    jcVar.j.setLocalPosition(new Vector3(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 0.02f));
                    ViewRenderable.builder().setView(jcVar.f4948h, R.layout.ground_plane_loader).build().thenAccept(new Consumer() { // from class: c.e.b.i3
                        @Override // java.util.function.Consumer
                        public final void accept(Object obj) {
                            jc jcVar2 = jc.this;
                            ViewRenderable viewRenderable = (ViewRenderable) obj;
                            Objects.requireNonNull(jcVar2);
                            ((ProgressBar) viewRenderable.getView().findViewById(R.id.imageTargetLoaderProgressbar)).setIndeterminate(true);
                            ((TextView) viewRenderable.getView().findViewById(R.id.imageTargetLoaderProgressbarText)).setText("");
                            viewRenderable.setShadowCaster(false);
                            viewRenderable.setShadowReceiver(false);
                            viewRenderable.setHorizontalAlignment(ViewRenderable.HorizontalAlignment.CENTER);
                            viewRenderable.setVerticalAlignment(ViewRenderable.VerticalAlignment.CENTER);
                            jcVar2.j.setRenderable(viewRenderable);
                            c.b.a.a.a.C(0.3f, 0.3f, 0.3f, jcVar2.j);
                        }
                    }).exceptionally(new Function() { // from class: c.e.b.u2
                        @Override // java.util.function.Function
                        public final Object apply(Object obj) {
                            Throwable th = (Throwable) obj;
                            Objects.requireNonNull(jc.this);
                            Log.e("LoaderARContentGroundPlaneSceneform", "Unable to load  renderable");
                            return null;
                        }
                    });
                    String string = Settings.Secure.getString(jcVar.f4947g.getContentResolver(), "android_id");
                    jcVar.f4946f = new ec();
                    String v = c.b.a.a.a.v(new StringBuilder(), jcVar.f4945e, "arcontent/get_ar_bundle_app_hash");
                    u.a aVar = new u.a();
                    aVar.c(f.u.f6108b);
                    aVar.a("campaign_hash", jcVar.f4944d);
                    aVar.a("scan_mode", "1");
                    aVar.a("device_id", string);
                    f.u b2 = aVar.b();
                    StringBuilder A = c.b.a.a.a.A(v, "  ");
                    A.append(b2.toString());
                    Log.d("LoaderARContentGroundPlaneSceneform", A.toString());
                    jcVar.f4946f.a(v, b2, new nc(jcVar, cbVar));
                    System.currentTimeMillis();
                }
                new Handler().postDelayed(new le(nonARCoreActivitySceneform), 500L);
            }
        });
        this.F = new p();
        this.B.getScene().getSunlight();
        this.D.setEnabled(false);
        this.q0 = new InstantTrackingHelper(this, this, (ViewGroup) findViewById(R.id.preview_display_layout), this.z.getSurface(), this.z.getSurfaceTexture(), new b());
        try {
            ActivityManager.MemoryInfo memoryInfo = new ActivityManager.MemoryInfo();
            ((ActivityManager) getSystemService("activity")).getMemoryInfo(memoryInfo);
            this.v0 = memoryInfo.totalMem / 1000000;
            Log.d("NonARCoreActivity", "RAM " + this.v0);
        } catch (Exception e2) {
            e2.printStackTrace();
            this.v0 = 4000L;
        }
        if (this.v0 < 3800) {
            Log.d("NonARCoreActivity", "Instant Tracking Retry interval increased");
            this.q0.setRetryInterval(2.0f);
        }
        this.q0.startTracking();
        Matrix.setIdentityM(this.s, 0);
        MediaPlayer.create(this, (int) R.raw.audio_screenshot);
        if (this.g0 == null && getIntent().hasExtra("fromargallery")) {
            A();
        }
    }

    @Override // b.b.c.h, b.q.b.d, android.app.Activity
    public void onDestroy() {
        this.q0.stopTracking();
        hd hdVar = this.J;
        if (hdVar != null) {
            hdVar.h();
        }
        super.onDestroy();
        Log.e("NonARCoreActivity", "onDestroy");
        Log.e("NonARCoreActivity", "memory free = " + Runtime.getRuntime().freeMemory());
        ((NotificationManager) this.x.getSystemService("notification")).cancel(10);
    }

    @Override // b.q.b.d, android.app.Activity, android.content.ComponentCallbacks
    public void onLowMemory() {
        super.onLowMemory();
        c.e.b.p000if.f.a(this.x, 60);
        Log.d("NonARCoreActivity", "Clear cached 3D models");
    }

    @Override // b.q.b.d, android.app.Activity
    public void onPause() {
        super.onPause();
        if (this.X) {
            this.m0 = false;
            this.b0.f4904c.pause();
        }
        InstantTrackingHelper instantTrackingHelper = this.q0;
        if (instantTrackingHelper != null) {
            instantTrackingHelper.onPause();
        }
        hd hdVar = this.J;
        if (hdVar != null) {
            hdVar.r();
        }
    }

    @Override // com.google.ar.sceneform.Scene.OnPeekTouchListener
    public void onPeekTouch(HitTestResult hitTestResult, MotionEvent motionEvent) {
        String str;
        int actionMasked = motionEvent.getActionMasked();
        if (actionMasked == 0) {
            this.t = motionEvent.getX();
            this.u = motionEvent.getY();
        } else if (actionMasked == 1) {
            float x = motionEvent.getX();
            if (this.u > motionEvent.getY() && this.U.getVisibility() == 0) {
                int[] iArr = new int[2];
                this.U.getLocationOnScreen(iArr);
                int i = iArr[0];
                Log.d("test36", this.t + "  " + x);
                float abs = Math.abs(this.t - x);
                if (this.u >= iArr[1] - 100 && abs < 220.0f) {
                    Iterator<c.e.b.hf.d> it = this.h0.iterator();
                    while (it.hasNext()) {
                        it.next().isSelected = false;
                    }
                    if (this.h0.size() > 0) {
                        this.h0.get(0).isSelected = true;
                        str = this.h0.get(0).name;
                    } else {
                        str = "";
                    }
                    this.k0.clear();
                    Iterator<c.e.b.hf.a> it2 = this.g0.iterator();
                    while (it2.hasNext()) {
                        c.e.b.hf.a next = it2.next();
                        if (next.category.equals(str)) {
                            this.k0.add(next);
                        }
                        if (new File(getCacheDir(), c.b.a.a.a.v(c.b.a.a.a.x("/assets/models/"), next.id, ".glb")).exists()) {
                            next.downloadStatus = 101;
                        }
                    }
                    this.i0 = new c.e.b.ef.c(this.h0, this);
                    this.d0 = new BottomSheetDialog(this);
                    View inflate = LayoutInflater.from(this).inflate(R.layout.ar_gallery_bottom_sheet, (ViewGroup) null);
                    this.e0 = (RecyclerView) inflate.findViewById(R.id.category_rec_view);
                    this.f0 = (RecyclerView) inflate.findViewById(R.id.thumbnails_rec_view);
                    this.e0.setAdapter(this.i0);
                    f fVar = new f(this.k0, this);
                    this.j0 = fVar;
                    this.f0.setAdapter(fVar);
                    this.d0.setContentView(inflate);
                    ((ConstraintLayout) inflate.findViewById(R.id.dismiss)).setOnClickListener(new ie(this));
                    this.d0.setOnDismissListener(new je(this));
                    this.d0.show();
                    this.U.setVisibility(4);
                }
            }
        }
        for (Node node : this.D.getChildren()) {
            if (node instanceof SimpleTransformableNode) {
                SimpleTransformableNode simpleTransformableNode = (SimpleTransformableNode) node;
                if (simpleTransformableNode.isSelected()) {
                    this.F.a(motionEvent, simpleTransformableNode);
                }
            }
        }
    }

    @Override // b.q.b.d, android.app.Activity
    public void onResume() {
        this.q0.onResume();
        hd hdVar = this.J;
        if (hdVar != null) {
            hdVar.v();
        }
        super.onResume();
        getWindow().getDecorView().setSystemUiVisibility(5894);
        if (this.X) {
            this.b0.f4904c.resume();
            this.m0 = true;
        }
    }

    @Override // com.google.ar.sceneform.Scene.OnUpdateListener
    public void onUpdate(FrameTime frameTime) {
    }

    @Override // android.app.Activity, android.view.Window.Callback
    public void onWindowFocusChanged(boolean z) {
        super.onWindowFocusChanged(z);
        if (z) {
            getWindow().getDecorView().setSystemUiVisibility(5894);
        }
    }

    public final void w() {
        Material.builder().setSource(this, R.raw.augmented_video_material).build().thenAccept(new Consumer() { // from class: c.e.b.hb
            @Override // java.util.function.Consumer
            public final void accept(Object obj) {
                NonARCoreActivitySceneform nonARCoreActivitySceneform = NonARCoreActivitySceneform.this;
                nonARCoreActivitySceneform.C.setRenderable(ShapeFactory.makeCube(new Vector3(6.4f, (nonARCoreActivitySceneform.s0 / nonARCoreActivitySceneform.r0) * 6.4f, 0.01f), Vector3.zero(), (Material) obj));
                nonARCoreActivitySceneform.C.getRenderable().getMaterial().setExternalTexture("videoTexture", nonARCoreActivitySceneform.z);
            }
        }).exceptionally(new Function() { // from class: c.e.b.jb
            @Override // java.util.function.Function
            public final Object apply(Object obj) {
                Throwable th = (Throwable) obj;
                Objects.requireNonNull(NonARCoreActivitySceneform.this);
                Log.e("NonARCoreActivity", "Unable to load camera renderable");
                return null;
            }
        });
    }

    public final void x(boolean z) {
        if (!z) {
            this.S.setVisibility(8);
        }
        if (z) {
            this.P.setImageDrawable(getDrawable(R.drawable.ic_video_recording));
            this.N.setVisibility(4);
            this.U.setVisibility(8);
            this.a0 = new l(this, this.B);
            return;
        }
        this.P.setImageDrawable(getDrawable(R.drawable.camerabutton));
        this.N.setVisibility(0);
        if (this.H != null) {
            this.U.setVisibility(0);
        }
        l lVar = this.a0;
        if (lVar != null) {
            lVar.a();
        }
    }

    public final void y(String str) {
        final hd hdVar = this.J;
        float[] fArr = {StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD};
        float[] fArr2 = {StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 0.707f, -0.707f, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD};
        hdVar.x = false;
        Node node = new Node();
        SimpleTransformableNode simpleTransformableNode = new SimpleTransformableNode(hdVar.l.getTransformationSystem());
        simpleTransformableNode.setParent(hdVar.i);
        simpleTransformableNode.setLocalPosition(new Vector3(fArr[0], fArr[1], fArr[2]));
        simpleTransformableNode.setLocalRotation(Quaternion.eulerAngles(new Vector3(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD)));
        node.setParent(simpleTransformableNode);
        node.setLocalRotation(Quaternion.multiply(Quaternion.eulerAngles(new Vector3(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 180.0f)), new Quaternion(fArr2[0], fArr2[1], fArr2[2], fArr2[3])));
        final Node node2 = new Node();
        node2.setParent(node);
        node2.setLocalRotation(Quaternion.eulerAngles(new Vector3(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 180.0f, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD)));
        hdVar.l.getArSceneView().getScene().addOnPeekTouchListener(new Scene.OnPeekTouchListener() { // from class: c.e.b.y5
            @Override // com.google.ar.sceneform.Scene.OnPeekTouchListener
            public final void onPeekTouch(HitTestResult hitTestResult, MotionEvent motionEvent) {
                hd.this.l.getTransformationSystem().onTouch(hitTestResult, motionEvent);
            }
        });
        simpleTransformableNode.getScaleController().setMinScale(0.01f);
        simpleTransformableNode.getScaleController().setMaxScale(0.07f);
        Log.d("LoaderARContent", "load3Dmodel model uri " + Uri.fromFile(new File(str)));
        ModelRenderable.builder().setSource(hdVar.f4816g, Uri.fromFile(new File(str))).setIsFilamentGltf(true).build().thenAccept(new Consumer() { // from class: c.e.b.z6
            @Override // java.util.function.Consumer
            public final void accept(Object obj) {
                hd hdVar2 = hd.this;
                Node node3 = node2;
                ve veVar = r3;
                ModelRenderable modelRenderable = (ModelRenderable) obj;
                Objects.requireNonNull(hdVar2);
                Log.d("LoaderARContent", "load3Dmodel model loaded");
                node3.setRenderable(modelRenderable).animate(true);
                if (node3.getRenderableInstance().getFilamentAsset().getAnimator().getAnimationCount() > 0) {
                    hdVar2.f(node3.getRenderableInstance(), "REPEAT", 1, "ALL", true, "model");
                }
                node3.setRenderable(modelRenderable);
                hdVar2.w(node3, veVar);
            }
        }).exceptionally(new Function() { // from class: c.e.b.f7
            @Override // java.util.function.Function
            public final Object apply(Object obj) {
                Objects.requireNonNull(hd.this);
                StringBuilder sb = new StringBuilder();
                sb.append("load3Dmodel ");
                c.b.a.a.a.N((Throwable) obj, sb, "LoaderARContent");
                return null;
            }
        });
        new Handler().postDelayed(new c(), 500L);
    }

    public void z(String str, boolean z) {
        Intent intent = new Intent(this.x, CapturePreview.class);
        if (z) {
            intent.putExtra("videoUrl", str);
        } else {
            intent.putExtra("imageUrl", str);
        }
        startActivity(intent);
    }
}