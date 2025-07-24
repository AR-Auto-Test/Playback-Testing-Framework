package c.e.b;

import android.animation.Animator;
import android.animation.AnimatorListenerAdapter;
import android.animation.ObjectAnimator;
import android.animation.ValueAnimator;
import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.graphics.Color;
import android.graphics.PorterDuff;
import android.graphics.Typeface;
import android.media.MediaPlayer;
import android.net.Uri;
import android.os.Handler;
import android.provider.Settings;
import android.util.Log;
import android.view.MotionEvent;
import android.view.animation.BounceInterpolator;
import android.view.animation.LinearInterpolator;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.TextView;
import c.e.b.hd;
import com.google.android.filament.Box;
import com.google.android.filament.gltfio.FilamentAsset;
import com.google.android.gms.common.internal.ImagesContract;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.google.ar.sceneform.HitTestResult;
import com.google.ar.sceneform.Node;
import com.google.ar.sceneform.Scene;
import com.google.ar.sceneform.math.Quaternion;
import com.google.ar.sceneform.math.Vector3;
import com.google.ar.sceneform.math.Vector3Evaluator;
import com.google.ar.sceneform.rendering.ExternalTexture;
import com.google.ar.sceneform.rendering.Light;
import com.google.ar.sceneform.rendering.Material;
import com.google.ar.sceneform.rendering.ModelRenderable;
import com.google.ar.sceneform.rendering.Renderable;
import com.google.ar.sceneform.rendering.RenderableInstance;
import com.google.ar.sceneform.rendering.ShapeFactory;
import com.google.ar.sceneform.rendering.Texture;
import com.google.ar.sceneform.rendering.ViewRenderable;
import com.google.ar.sceneform.ux.SimpleFragment;
import com.google.ar.sceneform.ux.SimpleTransformableNode;
import com.google.firebase.crashlytics.internal.common.CrashlyticsReportDataCapture;
import com.ibosoninnov.unitear.Player360Activity;
import com.ibosoninnov.unitear.R;
import com.ibosoninnov.unitear.VideoActivity;
import com.ibosoninnov.unitear.YoutubeView;
import f.u;
import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.NavigableMap;
import java.util.Objects;
import java.util.Timer;
import java.util.concurrent.CompletableFuture;
import java.util.function.BiConsumer;
import java.util.function.Consumer;
import java.util.function.Function;
import org.json.JSONException;
import org.json.JSONObject;

/* compiled from: LoaderARContentSceneform.java */
/* loaded from: classes2.dex */
public class hd {

    /* renamed from: a  reason: collision with root package name */
    public static MediaPlayer f4810a;

    /* renamed from: b  reason: collision with root package name */
    public static MediaPlayer f4811b;

    /* renamed from: c  reason: collision with root package name */
    public static MediaPlayer f4812c;

    /* renamed from: d  reason: collision with root package name */
    public static MediaPlayer f4813d;

    /* renamed from: e  reason: collision with root package name */
    public static MediaPlayer f4814e;
    public final String A;
    public String B;
    public boolean D;
    public final Map<Integer, String> N;
    public Map<Integer, NavigableMap> O;
    public ImageView P;
    public ProgressBar Q;
    public TextView R;
    public TextView S;
    public ObjectAnimator T;
    public float U;

    /* renamed from: f  reason: collision with root package name */
    public final Context f4815f;

    /* renamed from: g  reason: collision with root package name */
    public final Activity f4816g;

    /* renamed from: h  reason: collision with root package name */
    public cc f4817h;
    public final Node i;
    public Node j;
    public g k;
    public final SimpleFragment l;
    public c.e.b.p000if.e m;
    public final ArrayList<Node> n;
    public final Handler o;
    public final Runnable p;
    public long q;
    public int r;
    public int s;
    public int t;
    public boolean v;
    public boolean w;
    public final String y;
    public final String z;
    public boolean u = true;
    public boolean x = false;
    public String C = CrashlyticsReportDataCapture.SIGNAL_DEFAULT;
    public final Map<Integer, ObjectAnimator> E = new HashMap();
    public final Map<Integer, String[]> F = new HashMap();
    public final Map<Integer, String> G = new HashMap();
    public final Map<Integer, Node> H = new HashMap();
    public final Map<Integer, ve> I = new HashMap();
    public final Map<MediaPlayer, Boolean> J = new HashMap();
    public final Map<MediaPlayer, Boolean> K = new HashMap();
    public final Map<Node, ve> L = new HashMap();
    public final Map<Integer, ModelRenderable> M = new HashMap();

    /* compiled from: LoaderARContentSceneform.java */
    /* loaded from: classes2.dex */
    public class a extends AnimatorListenerAdapter {

        /* renamed from: a  reason: collision with root package name */
        public final /* synthetic */ int f4818a;

        public a(int i) {
            this.f4818a = i;
        }

        @Override // android.animation.AnimatorListenerAdapter, android.animation.Animator.AnimatorListener
        public void onAnimationEnd(Animator animator) {
            super.onAnimationEnd(animator);
            ve veVar = hd.this.I.get(Integer.valueOf(this.f4818a));
            Objects.requireNonNull(veVar);
            if (Objects.equals(veVar.A, "REPEAT")) {
                hd hdVar = hd.this;
                Node node = hdVar.H.get(Integer.valueOf(this.f4818a));
                Objects.requireNonNull(node);
                RenderableInstance renderableInstance = node.getRenderableInstance();
                ve veVar2 = hd.this.I.get(Integer.valueOf(this.f4818a));
                Objects.requireNonNull(veVar2);
                int i = veVar2.H;
                ve veVar3 = hd.this.I.get(Integer.valueOf(this.f4818a));
                Objects.requireNonNull(veVar3);
                String str = veVar3.z;
                ve veVar4 = hd.this.I.get(Integer.valueOf(this.f4818a));
                Objects.requireNonNull(veVar4);
                hdVar.f(renderableInstance, "REPEAT", i, str, veVar4.j, "model");
                return;
            }
            ve veVar5 = hd.this.I.get(Integer.valueOf(this.f4818a));
            Objects.requireNonNull(veVar5);
            if (Objects.equals(veVar5.A, "REPEAT_ONCE")) {
                hd hdVar2 = hd.this;
                Node node2 = hdVar2.H.get(Integer.valueOf(this.f4818a));
                Objects.requireNonNull(node2);
                RenderableInstance renderableInstance2 = node2.getRenderableInstance();
                ve veVar6 = hd.this.I.get(Integer.valueOf(this.f4818a));
                Objects.requireNonNull(veVar6);
                int i2 = veVar6.H;
                ve veVar7 = hd.this.I.get(Integer.valueOf(this.f4818a));
                Objects.requireNonNull(veVar7);
                String str2 = veVar7.z;
                ve veVar8 = hd.this.I.get(Integer.valueOf(this.f4818a));
                Objects.requireNonNull(veVar8);
                hdVar2.f(renderableInstance2, "REPEAT_ONCE", i2, str2, veVar8.j, "model");
            }
        }
    }

    /* compiled from: LoaderARContentSceneform.java */
    /* loaded from: classes2.dex */
    public class b extends AnimatorListenerAdapter {

        /* renamed from: a  reason: collision with root package name */
        public final /* synthetic */ int f4820a;

        /* renamed from: b  reason: collision with root package name */
        public final /* synthetic */ Node f4821b;

        public b(int i, Node node) {
            this.f4820a = i;
            this.f4821b = node;
        }

        @Override // android.animation.AnimatorListenerAdapter, android.animation.Animator.AnimatorListener
        public void onAnimationEnd(Animator animator) {
            super.onAnimationEnd(animator);
            if (this.f4820a == 1) {
                hd.this.s(this.f4821b);
            }
        }
    }

    /* compiled from: LoaderARContentSceneform.java */
    /* loaded from: classes2.dex */
    public class c implements Animator.AnimatorListener {
        public c() {
        }

        @Override // android.animation.Animator.AnimatorListener
        public void onAnimationCancel(Animator animator) {
        }

        @Override // android.animation.Animator.AnimatorListener
        public void onAnimationEnd(Animator animator) {
            Node node = hd.this.j;
            if (node != null) {
                node.setParent(null);
                hd.this.j = null;
            }
            hd.this.L.forEach(new BiConsumer() { // from class: c.e.b.j4
                @Override // java.util.function.BiConsumer
                public final void accept(Object obj, Object obj2) {
                    hd.c cVar = hd.c.this;
                    Node node2 = (Node) obj;
                    Objects.requireNonNull(cVar);
                    node2.setEnabled(true);
                    hd.this.e(node2, (ve) obj2, 0);
                }
            });
            if (hd.this.i.getChildren().size() == 0) {
                hd.this.k.a("No contents");
                return;
            }
            hd hdVar = hd.this;
            Objects.requireNonNull(hdVar);
            try {
                hdVar.J.forEach(w5.f5371a);
            } catch (IllegalStateException e2) {
                PrintStream printStream = System.err;
                StringBuilder x = c.b.a.a.a.x("An IllegalStateException occurred: ");
                x.append(e2.getMessage());
                printStream.println(x.toString());
                e2.printStackTrace();
            }
        }

        @Override // android.animation.Animator.AnimatorListener
        public void onAnimationRepeat(Animator animator) {
        }

        @Override // android.animation.Animator.AnimatorListener
        public void onAnimationStart(Animator animator) {
        }
    }

    /* compiled from: LoaderARContentSceneform.java */
    /* loaded from: classes2.dex */
    public class d implements c.e.b.gf.c {

        /* renamed from: a  reason: collision with root package name */
        public final /* synthetic */ ve f4824a;

        /* renamed from: b  reason: collision with root package name */
        public final /* synthetic */ String f4825b;

        /* renamed from: c  reason: collision with root package name */
        public final /* synthetic */ Node f4826c;

        /* renamed from: d  reason: collision with root package name */
        public final /* synthetic */ Node f4827d;

        /* renamed from: e  reason: collision with root package name */
        public final /* synthetic */ SimpleTransformableNode f4828e;

        public d(ve veVar, String str, Node node, Node node2, SimpleTransformableNode simpleTransformableNode) {
            this.f4824a = veVar;
            this.f4825b = str;
            this.f4826c = node;
            this.f4827d = node2;
            this.f4828e = simpleTransformableNode;
        }

        @Override // c.e.b.gf.c
        public void a(String str, int i, String str2) {
        }

        @Override // c.e.b.gf.c
        public void b(String str, String str2) {
            ve veVar = this.f4824a;
            int i = veVar.f5352c;
            if (i != 0) {
                String str3 = (String) veVar.T.get(Integer.valueOf(i));
                CompletableFuture<ModelRenderable> build = ModelRenderable.builder().setSource(hd.this.f4816g, Uri.parse(this.f4825b)).setIsFilamentGltf(true).build();
                final ve veVar2 = this.f4824a;
                final Node node = this.f4826c;
                final Node node2 = this.f4827d;
                CompletableFuture<Void> thenAccept = build.thenAccept(new Consumer() { // from class: c.e.b.b5
                    @Override // java.util.function.Consumer
                    public final void accept(Object obj) {
                        hd.d dVar = hd.d.this;
                        ve veVar3 = veVar2;
                        Node node3 = node;
                        Node node4 = node2;
                        ModelRenderable modelRenderable = (ModelRenderable) obj;
                        hd.this.M.put(Integer.valueOf(veVar3.H), modelRenderable.makeCopy());
                        node3.setRenderable(modelRenderable);
                        hd hdVar = hd.this;
                        hdVar.p(hdVar.M.get(Integer.valueOf(veVar3.H)), node3, (String) veVar3.T.get(Integer.valueOf(veVar3.f5352c)), veVar3.H);
                        hd.this.f(node3.getRenderableInstance(), veVar3.A, veVar3.H, veVar3.z, veVar3.j, "model");
                        hd.this.E.put(Integer.valueOf(veVar3.H), hd.this.T);
                        hd.this.w(node3, veVar3);
                        hd.this.H.put(Integer.valueOf(veVar3.H), node3);
                        hd.this.d(veVar3);
                        hd.this.L.put(node4, veVar3);
                        node4.setEnabled(false);
                        hd hdVar2 = hd.this;
                        int i2 = hdVar2.r - 1;
                        hdVar2.r = i2;
                        if (i2 == 0) {
                            hdVar2.k();
                            return;
                        }
                        StringBuilder x = c.b.a.a.a.x("Your AR content is loading  (");
                        hd hdVar3 = hd.this;
                        c.b.a.a.a.D(hdVar3.s, hdVar3.r, x, "/");
                        hd.this.R.setText(c.b.a.a.a.s(x, hd.this.s, ")"));
                    }
                });
                final SimpleTransformableNode simpleTransformableNode = this.f4828e;
                thenAccept.exceptionally(new Function() { // from class: c.e.b.z4
                    @Override // java.util.function.Function
                    public final Object apply(Object obj) {
                        hd.d dVar = hd.d.this;
                        SimpleTransformableNode simpleTransformableNode2 = simpleTransformableNode;
                        Objects.requireNonNull(dVar);
                        StringBuilder sb = new StringBuilder();
                        sb.append("load3Dmodel ");
                        c.b.a.a.a.N((Throwable) obj, sb, "LoaderARContent");
                        hd.this.i.removeChild(simpleTransformableNode2);
                        hd hdVar = hd.this;
                        int i2 = hdVar.r - 1;
                        hdVar.r = i2;
                        if (i2 == 0) {
                            hdVar.k();
                            return null;
                        }
                        return null;
                    }
                });
                return;
            }
            CompletableFuture<ModelRenderable> build2 = ModelRenderable.builder().setSource(hd.this.f4816g, Uri.parse(this.f4825b)).setIsFilamentGltf(true).build();
            final ve veVar3 = this.f4824a;
            final Node node3 = this.f4826c;
            final Node node4 = this.f4827d;
            CompletableFuture<Void> thenAccept2 = build2.thenAccept(new Consumer() { // from class: c.e.b.a5
                @Override // java.util.function.Consumer
                public final void accept(Object obj) {
                    hd.d dVar = hd.d.this;
                    ve veVar4 = veVar3;
                    Node node5 = node3;
                    Node node6 = node4;
                    ModelRenderable modelRenderable = (ModelRenderable) obj;
                    hd.this.M.put(Integer.valueOf(veVar4.H), modelRenderable.makeCopy());
                    hd.this.N.put(Integer.valueOf(veVar4.H), "default");
                    node5.setRenderable(modelRenderable);
                    hd.this.f(node5.getRenderableInstance(), veVar4.A, veVar4.H, veVar4.z, veVar4.j, "model");
                    hd.this.w(node5, veVar4);
                    hd.this.H.put(Integer.valueOf(veVar4.H), node5);
                    hd.this.d(veVar4);
                    hd.this.L.put(node6, veVar4);
                    node6.setEnabled(false);
                    hd hdVar = hd.this;
                    int i2 = hdVar.r - 1;
                    hdVar.r = i2;
                    if (i2 == 0) {
                        hdVar.k();
                        return;
                    }
                    StringBuilder x = c.b.a.a.a.x("Your AR content is loading  (");
                    hd hdVar2 = hd.this;
                    c.b.a.a.a.D(hdVar2.s, hdVar2.r, x, "/");
                    hd.this.R.setText(c.b.a.a.a.s(x, hd.this.s, ")"));
                }
            });
            final SimpleTransformableNode simpleTransformableNode2 = this.f4828e;
            thenAccept2.exceptionally(new Function() { // from class: c.e.b.y4
                @Override // java.util.function.Function
                public final Object apply(Object obj) {
                    hd.d dVar = hd.d.this;
                    SimpleTransformableNode simpleTransformableNode3 = simpleTransformableNode2;
                    Objects.requireNonNull(dVar);
                    StringBuilder sb = new StringBuilder();
                    sb.append("load3Dmodel ");
                    c.b.a.a.a.N((Throwable) obj, sb, "LoaderARContent");
                    hd.this.i.removeChild(simpleTransformableNode3);
                    hd hdVar = hd.this;
                    int i2 = hdVar.r - 1;
                    hdVar.r = i2;
                    if (i2 == 0) {
                        hdVar.k();
                        return null;
                    }
                    return null;
                }
            });
        }
    }

    /* compiled from: LoaderARContentSceneform.java */
    /* loaded from: classes2.dex */
    public class e extends AnimatorListenerAdapter {

        /* renamed from: a  reason: collision with root package name */
        public final /* synthetic */ int f4830a;

        /* renamed from: b  reason: collision with root package name */
        public final /* synthetic */ int[] f4831b;

        /* renamed from: c  reason: collision with root package name */
        public final /* synthetic */ int f4832c;

        /* renamed from: d  reason: collision with root package name */
        public final /* synthetic */ String f4833d;

        /* renamed from: e  reason: collision with root package name */
        public final /* synthetic */ RenderableInstance f4834e;

        /* renamed from: f  reason: collision with root package name */
        public final /* synthetic */ String f4835f;

        public e(int i, int[] iArr, int i2, String str, RenderableInstance renderableInstance, String str2) {
            this.f4830a = i;
            this.f4831b = iArr;
            this.f4832c = i2;
            this.f4833d = str;
            this.f4834e = renderableInstance;
            this.f4835f = str2;
        }

        @Override // android.animation.AnimatorListenerAdapter, android.animation.Animator.AnimatorListener
        public void onAnimationEnd(Animator animator) {
            super.onAnimationEnd(animator);
            hd hdVar = hd.this;
            hdVar.T = hdVar.E.get(Integer.valueOf(this.f4830a));
            int[] iArr = this.f4831b;
            iArr[0] = iArr[0] + 1;
            if (iArr[0] == this.f4832c) {
                if (this.f4833d.equals("REPEAT")) {
                    int[] iArr2 = this.f4831b;
                    iArr2[0] = 0;
                    hd.this.T = this.f4834e.animate(iArr2[0]);
                    hd.this.T.setRepeatCount(0);
                    hd.this.T.addListener(this);
                    hd.this.T.start();
                    hd.this.E.put(Integer.valueOf(this.f4830a), hd.this.T);
                    return;
                } else if (Objects.equals(this.f4835f, "trigger")) {
                    hd.this.T.pause();
                    hd.this.T.removeAllUpdateListeners();
                    hd.this.T.removeAllListeners();
                    hd.this.T.end();
                    hd.this.T.cancel();
                    ve veVar = hd.this.I.get(Integer.valueOf(this.f4830a));
                    Objects.requireNonNull(veVar);
                    if (Objects.equals(veVar.A, "REPEAT")) {
                        hd hdVar2 = hd.this;
                        Node node = hdVar2.H.get(Integer.valueOf(this.f4830a));
                        Objects.requireNonNull(node);
                        RenderableInstance renderableInstance = node.getRenderableInstance();
                        ve veVar2 = hd.this.I.get(Integer.valueOf(this.f4830a));
                        Objects.requireNonNull(veVar2);
                        int i = veVar2.H;
                        ve veVar3 = hd.this.I.get(Integer.valueOf(this.f4830a));
                        Objects.requireNonNull(veVar3);
                        String str = veVar3.z;
                        ve veVar4 = hd.this.I.get(Integer.valueOf(this.f4830a));
                        Objects.requireNonNull(veVar4);
                        hdVar2.f(renderableInstance, "REPEAT", i, str, veVar4.j, "model");
                        return;
                    }
                    ve veVar5 = hd.this.I.get(Integer.valueOf(this.f4830a));
                    Objects.requireNonNull(veVar5);
                    if (Objects.equals(veVar5.A, "REPEAT_ONCE")) {
                        hd hdVar3 = hd.this;
                        Node node2 = hdVar3.H.get(Integer.valueOf(this.f4830a));
                        Objects.requireNonNull(node2);
                        RenderableInstance renderableInstance2 = node2.getRenderableInstance();
                        ve veVar6 = hd.this.I.get(Integer.valueOf(this.f4830a));
                        Objects.requireNonNull(veVar6);
                        int i2 = veVar6.H;
                        ve veVar7 = hd.this.I.get(Integer.valueOf(this.f4830a));
                        Objects.requireNonNull(veVar7);
                        String str2 = veVar7.z;
                        ve veVar8 = hd.this.I.get(Integer.valueOf(this.f4830a));
                        Objects.requireNonNull(veVar8);
                        hdVar3.f(renderableInstance2, "REPEAT_ONCE", i2, str2, veVar8.j, "model");
                        return;
                    }
                    return;
                } else {
                    hd.this.T.pause();
                    hd.this.T.removeAllListeners();
                    hd.this.T.setPropertyName("Idle");
                    hd.this.E.put(Integer.valueOf(this.f4830a), hd.this.T);
                    return;
                }
            }
            hd.this.T = this.f4834e.animate(iArr[0]);
            hd.this.T.setRepeatCount(0);
            hd.this.T.addListener(this);
            hd.this.T.start();
            hd.this.E.put(Integer.valueOf(this.f4830a), hd.this.T);
        }
    }

    /* compiled from: LoaderARContentSceneform.java */
    /* loaded from: classes2.dex */
    public interface f {
    }

    /* compiled from: LoaderARContentSceneform.java */
    /* loaded from: classes2.dex */
    public interface g {
        void a(String str);
    }

    @SuppressLint({"HardwareIds"})
    public hd(String str, String str2, Node node, SimpleFragment simpleFragment, Context context, Activity activity) {
        new HashMap();
        this.N = new HashMap();
        this.O = new HashMap();
        this.Q = null;
        this.S = null;
        this.y = str;
        this.z = str2;
        this.i = node;
        this.l = simpleFragment;
        this.f4815f = context;
        this.f4816g = activity;
        this.n = new ArrayList<>();
        this.o = new Handler();
        this.p = new Runnable() { // from class: c.e.b.o6
            @Override // java.lang.Runnable
            public final void run() {
                hd.this.z(false);
            }
        };
        this.A = Settings.Secure.getString(context.getContentResolver(), "android_id");
        this.m = new c.e.b.p000if.e(context);
    }

    public static void a(final hd hdVar, JSONObject jSONObject, int i) {
        Node node = hdVar.j;
        if (node != null) {
            ViewRenderable viewRenderable = (ViewRenderable) node.getRenderable();
            hdVar.Q = (ProgressBar) viewRenderable.getView().findViewById(R.id.imageTargetLoaderProgressbar);
            hdVar.R = (TextView) viewRenderable.getView().findViewById(R.id.tagetcontent);
            hdVar.S = (TextView) viewRenderable.getView().findViewById(R.id.imageTargetLoaderProgressbarText);
        }
        try {
            int i2 = jSONObject.getInt("contentTypeId");
            try {
                Integer.parseInt(jSONObject.getString("id"));
                final ve veVar = new ve(jSONObject, i2);
                hdVar.O.put(Integer.valueOf(veVar.H), veVar.T);
                switch (i2) {
                    case 1:
                        hdVar.f4816g.runOnUiThread(new Runnable() { // from class: c.e.b.j5
                            @Override // java.lang.Runnable
                            public final void run() {
                                final hd hdVar2 = hd.this;
                                final ve veVar2 = veVar;
                                Objects.requireNonNull(hdVar2);
                                float f2 = veVar2.m[1];
                                final SimpleTransformableNode simpleTransformableNode = new SimpleTransformableNode(hdVar2.l.getTransformationSystem());
                                simpleTransformableNode.setParent(hdVar2.i);
                                float[] fArr = veVar2.l;
                                final Node Q = c.b.a.a.a.Q(simpleTransformableNode, new Vector3(fArr[0], fArr[1], fArr[2]), simpleTransformableNode);
                                float[] fArr2 = veVar2.q;
                                Q.setLocalRotation(new Quaternion(-fArr2[0], -fArr2[1], fArr2[2], fArr2[3]));
                                float[] fArr3 = veVar2.n;
                                if (fArr3[0] != StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                                    Q.setLocalScale(new Vector3(fArr3[0], fArr3[1], fArr3[2]).scaled(veVar2.p));
                                }
                                hdVar2.l.getArSceneView().getScene().addOnPeekTouchListener(new Scene.OnPeekTouchListener() { // from class: c.e.b.f4
                                    @Override // com.google.ar.sceneform.Scene.OnPeekTouchListener
                                    public final void onPeekTouch(HitTestResult hitTestResult, MotionEvent motionEvent) {
                                        hd.this.l.getTransformationSystem().onTouch(hitTestResult, motionEvent);
                                    }
                                });
                                c.b.a.a.a.c(f2, 0.01f, simpleTransformableNode.getScaleController(), simpleTransformableNode).setMaxScale(f2 * 0.08f);
                                final MediaPlayer m = hdVar2.m();
                                m.setOnCompletionListener(new MediaPlayer.OnCompletionListener() { // from class: c.e.b.e6
                                    @Override // android.media.MediaPlayer.OnCompletionListener
                                    public final void onCompletion(MediaPlayer mediaPlayer) {
                                        hd hdVar3 = hd.this;
                                        ve veVar3 = veVar2;
                                        Objects.requireNonNull(hdVar3);
                                        if (Objects.equals(veVar3.f5355f, "GOTO_SCENE_AFTER_CONTENT_END")) {
                                            hdVar3.C = veVar3.f5356g;
                                            hdVar3.n();
                                        }
                                    }
                                });
                                final ExternalTexture externalTexture = new ExternalTexture();
                                m.setSurface(externalTexture.getSurface());
                                m.setAudioStreamType(3);
                                try {
                                    m.setScreenOnWhilePlaying(true);
                                    m.setDataSource(veVar2.f5354e);
                                    m.prepareAsync();
                                    m.setOnPreparedListener(new MediaPlayer.OnPreparedListener() { // from class: c.e.b.q6
                                        @Override // android.media.MediaPlayer.OnPreparedListener
                                        public final void onPrepared(MediaPlayer mediaPlayer) {
                                            final hd hdVar3 = hd.this;
                                            final ve veVar3 = veVar2;
                                            final MediaPlayer mediaPlayer2 = m;
                                            final ExternalTexture externalTexture2 = externalTexture;
                                            final Node node2 = Q;
                                            final SimpleTransformableNode simpleTransformableNode2 = simpleTransformableNode;
                                            Objects.requireNonNull(hdVar3);
                                            final float videoHeight = mediaPlayer.getVideoHeight();
                                            final float videoWidth = mediaPlayer.getVideoWidth();
                                            int i3 = veVar3.u ? R.raw.chroma_key_video_material : R.raw.augmented_video_material;
                                            if (veVar3.o) {
                                                mediaPlayer2.setLooping(true);
                                            }
                                            Material.builder().setSource(hdVar3.f4816g, i3).build().thenAccept(new Consumer() { // from class: c.e.b.x6
                                                @Override // java.util.function.Consumer
                                                public final void accept(Object obj) {
                                                    boolean z;
                                                    final hd hdVar4 = hd.this;
                                                    final ve veVar4 = veVar3;
                                                    float f3 = videoWidth;
                                                    float f4 = videoHeight;
                                                    ExternalTexture externalTexture3 = externalTexture2;
                                                    Node node3 = node2;
                                                    final MediaPlayer mediaPlayer3 = mediaPlayer2;
                                                    Material material = (Material) obj;
                                                    Objects.requireNonNull(hdVar4);
                                                    if (veVar4.u) {
                                                        float[] fArr4 = veVar4.y;
                                                        z = false;
                                                        material.setFloat4("keyColor", fArr4[0], fArr4[1], fArr4[2], 1.0f);
                                                    } else {
                                                        z = false;
                                                    }
                                                    float f5 = f3 / f4;
                                                    ModelRenderable makeCube = ShapeFactory.makeCube(new Vector3(f5 * 4.4f, 4.4f, 1.0E-4f), Vector3.zero(), material);
                                                    makeCube.setShadowCaster(z);
                                                    makeCube.setShadowReceiver(z);
                                                    makeCube.getMaterial().setExternalTexture("videoTexture", externalTexture3);
                                                    node3.setRenderable(makeCube);
                                                    final Node node4 = new Node();
                                                    node4.setName("playPauseButton");
                                                    hdVar4.n.add(node4);
                                                    node4.setParent(node3);
                                                    node4.setLocalPosition(new Vector3(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 0.001f));
                                                    node4.setOnAccurateTapListner(new Node.OnAccurateTapListner() { // from class: c.e.b.r5
                                                        @Override // com.google.ar.sceneform.Node.OnAccurateTapListner
                                                        public final void onAccurateTap(HitTestResult hitTestResult, MotionEvent motionEvent) {
                                                            Node node5 = Node.this;
                                                            MediaPlayer mediaPlayer4 = mediaPlayer3;
                                                            ImageView imageView = (ImageView) ((ViewRenderable) node5.getRenderable()).getView().findViewById(R.id.img_loader_view);
                                                            if (mediaPlayer4.isPlaying()) {
                                                                mediaPlayer4.pause();
                                                                imageView.setImageResource(R.drawable.play);
                                                                return;
                                                            }
                                                            mediaPlayer4.start();
                                                            imageView.setImageResource(R.drawable.pause);
                                                        }
                                                    });
                                                    node3.setOnAccurateTapListner(new Node.OnAccurateTapListner() { // from class: c.e.b.k6
                                                        @Override // com.google.ar.sceneform.Node.OnAccurateTapListner
                                                        public final void onAccurateTap(HitTestResult hitTestResult, MotionEvent motionEvent) {
                                                            hd hdVar5 = hd.this;
                                                            ve veVar5 = veVar4;
                                                            MediaPlayer mediaPlayer4 = mediaPlayer3;
                                                            Node node5 = node4;
                                                            Objects.requireNonNull(hdVar5);
                                                            String str = veVar5.f5355f;
                                                            str.hashCode();
                                                            char c2 = 65535;
                                                            switch (str.hashCode()) {
                                                                case -1858525552:
                                                                    if (str.equals("GOTO_SCENE")) {
                                                                        c2 = 0;
                                                                        break;
                                                                    }
                                                                    break;
                                                                case -1502373865:
                                                                    if (str.equals("GOTO_FULLSCREEN")) {
                                                                        c2 = 1;
                                                                        break;
                                                                    }
                                                                    break;
                                                                case -1141634779:
                                                                    if (str.equals("PLAY_PAUSE_CONTENT")) {
                                                                        c2 = 2;
                                                                        break;
                                                                    }
                                                                    break;
                                                                case 827942371:
                                                                    if (str.equals("GOTO_SCENE_AFTER_CONTENT_END")) {
                                                                        c2 = 3;
                                                                        break;
                                                                    }
                                                                    break;
                                                            }
                                                            switch (c2) {
                                                                case 0:
                                                                    hdVar5.C = veVar5.f5356g;
                                                                    hdVar5.n();
                                                                    return;
                                                                case 1:
                                                                    Intent intent = new Intent(hdVar5.f4815f, VideoActivity.class);
                                                                    intent.putExtra("videoUrl", veVar5.f5354e);
                                                                    intent.putExtra("loop", veVar5.o);
                                                                    intent.putExtra("currenttime", mediaPlayer4.getCurrentPosition());
                                                                    hdVar5.f4815f.startActivity(intent);
                                                                    return;
                                                                case 2:
                                                                case 3:
                                                                    ImageView imageView = (ImageView) ((ViewRenderable) node5.getRenderable()).getView().findViewById(R.id.img_loader_view);
                                                                    if (mediaPlayer4.isPlaying()) {
                                                                        mediaPlayer4.pause();
                                                                        imageView.setImageResource(R.drawable.play);
                                                                        return;
                                                                    }
                                                                    mediaPlayer4.start();
                                                                    imageView.setImageResource(R.drawable.pause);
                                                                    return;
                                                                default:
                                                                    return;
                                                            }
                                                        }
                                                    });
                                                    ViewRenderable.builder().setView(hdVar4.f4816g, R.layout.img_view).build().thenAccept(new Consumer() { // from class: c.e.b.k5
                                                        @Override // java.util.function.Consumer
                                                        public final void accept(Object obj2) {
                                                            ve veVar5 = ve.this;
                                                            Node node5 = node4;
                                                            ViewRenderable viewRenderable2 = (ViewRenderable) obj2;
                                                            viewRenderable2.setHorizontalAlignment(ViewRenderable.HorizontalAlignment.CENTER);
                                                            viewRenderable2.setVerticalAlignment(ViewRenderable.VerticalAlignment.CENTER);
                                                            ImageView imageView = (ImageView) viewRenderable2.getView().findViewById(R.id.img_loader_view);
                                                            if (veVar5.j) {
                                                                imageView.setImageResource(R.drawable.pause);
                                                            } else {
                                                                imageView.setImageResource(R.drawable.play);
                                                            }
                                                            node5.setRenderable(viewRenderable2);
                                                            node5.setLocalScale(new Vector3(1.0f, 1.0f, 1.0f));
                                                        }
                                                    }).exceptionally(new Function() { // from class: c.e.b.d5
                                                        @Override // java.util.function.Function
                                                        public final Object apply(Object obj2) {
                                                            hd hdVar5 = hd.this;
                                                            Throwable th = (Throwable) obj2;
                                                            Objects.requireNonNull(hdVar5);
                                                            Log.e("LoaderARContent", "Unable to load  play button");
                                                            int i4 = hdVar5.r - 1;
                                                            hdVar5.r = i4;
                                                            if (i4 == 0) {
                                                                hdVar5.k();
                                                                return null;
                                                            }
                                                            return null;
                                                        }
                                                    });
                                                    final Node node5 = new Node();
                                                    hdVar4.n.add(node5);
                                                    node5.setParent(node3);
                                                    node5.setLocalPosition(new Vector3(f5 * 2.0f, -2.0f, 0.001f));
                                                    node5.setOnAccurateTapListner(new Node.OnAccurateTapListner() { // from class: c.e.b.g5
                                                        @Override // com.google.ar.sceneform.Node.OnAccurateTapListner
                                                        public final void onAccurateTap(HitTestResult hitTestResult, MotionEvent motionEvent) {
                                                            hd hdVar5 = hd.this;
                                                            ve veVar5 = veVar4;
                                                            Objects.requireNonNull(hdVar5);
                                                            Intent intent = new Intent(hdVar5.f4815f, VideoActivity.class);
                                                            intent.putExtra("videoUrl", veVar5.f5354e);
                                                            intent.putExtra("loop", veVar5.o);
                                                            hdVar5.f4815f.startActivity(intent);
                                                        }
                                                    });
                                                    ViewRenderable.builder().setView(hdVar4.f4816g, R.layout.img_view).build().thenAccept(new Consumer() { // from class: c.e.b.s6
                                                        @Override // java.util.function.Consumer
                                                        public final void accept(Object obj2) {
                                                            Node node6 = Node.this;
                                                            ViewRenderable viewRenderable2 = (ViewRenderable) obj2;
                                                            ((ImageView) viewRenderable2.getView().findViewById(R.id.img_loader_view)).setImageResource(R.drawable.fullscreen);
                                                            node6.setRenderable(viewRenderable2);
                                                            c.b.a.a.a.C(0.6f, 0.6f, 0.6f, node6);
                                                        }
                                                    }).exceptionally(new Function() { // from class: c.e.b.y6
                                                        @Override // java.util.function.Function
                                                        public final Object apply(Object obj2) {
                                                            hd hdVar5 = hd.this;
                                                            Throwable th = (Throwable) obj2;
                                                            Objects.requireNonNull(hdVar5);
                                                            Log.e("LoaderARContent", "Unable to load  fullscreen button");
                                                            int i4 = hdVar5.r - 1;
                                                            hdVar5.r = i4;
                                                            if (i4 == 0) {
                                                                hdVar5.k();
                                                                return null;
                                                            }
                                                            return null;
                                                        }
                                                    });
                                                    if (veVar4.j) {
                                                        hdVar4.J.put(mediaPlayer3, Boolean.TRUE);
                                                        hdVar4.c(10);
                                                    } else {
                                                        hdVar4.c(2000);
                                                    }
                                                    mediaPlayer3.seekTo(1);
                                                    hdVar4.L.put(node3, veVar4);
                                                    node3.setEnabled(z);
                                                    mediaPlayer3.setOnSeekCompleteListener(new MediaPlayer.OnSeekCompleteListener() { // from class: c.e.b.a6
                                                        @Override // android.media.MediaPlayer.OnSeekCompleteListener
                                                        public final void onSeekComplete(MediaPlayer mediaPlayer4) {
                                                            hd hdVar5 = hd.this;
                                                            int i4 = hdVar5.r - 1;
                                                            hdVar5.r = i4;
                                                            if (i4 == 0) {
                                                                hdVar5.k();
                                                                return;
                                                            }
                                                            StringBuilder x = c.b.a.a.a.x("Your AR content is loading  (");
                                                            c.b.a.a.a.D(hdVar5.s, hdVar5.r, x, "/");
                                                            hdVar5.R.setText(c.b.a.a.a.s(x, hdVar5.s, ")"));
                                                        }
                                                    });
                                                }
                                            }).exceptionally(new Function() { // from class: c.e.b.q5
                                                @Override // java.util.function.Function
                                                public final Object apply(Object obj) {
                                                    hd hdVar4 = hd.this;
                                                    SimpleTransformableNode simpleTransformableNode3 = simpleTransformableNode2;
                                                    Throwable th = (Throwable) obj;
                                                    Objects.requireNonNull(hdVar4);
                                                    Log.e("LoaderARContent", "Unable to load  video player");
                                                    hdVar4.i.removeChild(simpleTransformableNode3);
                                                    int i4 = hdVar4.r - 1;
                                                    hdVar4.r = i4;
                                                    if (i4 == 0) {
                                                        hdVar4.k();
                                                        return null;
                                                    }
                                                    return null;
                                                }
                                            });
                                        }
                                    });
                                } catch (IOException e2) {
                                    e2.printStackTrace();
                                }
                            }
                        });
                        break;
                    case 2:
                        hdVar.f4816g.runOnUiThread(new Runnable() { // from class: c.e.b.v4
                            @Override // java.lang.Runnable
                            public final void run() {
                                MediaPlayer mediaPlayer;
                                final hd hdVar2 = hd.this;
                                final ve veVar2 = veVar;
                                Objects.requireNonNull(hdVar2);
                                float f2 = veVar2.m[1];
                                final String a2 = new cf().a(veVar2.f5354e);
                                final String r = c.b.a.a.a.r("https://img.youtube.com/vi/", a2, "/hqdefault.jpg");
                                SimpleTransformableNode simpleTransformableNode = new SimpleTransformableNode(hdVar2.l.getTransformationSystem());
                                simpleTransformableNode.setParent(hdVar2.i);
                                float[] fArr = veVar2.l;
                                final Node Q = c.b.a.a.a.Q(simpleTransformableNode, new Vector3(fArr[0], fArr[1], fArr[2]), simpleTransformableNode);
                                float[] fArr2 = veVar2.q;
                                Q.setLocalRotation(new Quaternion(-fArr2[0], -fArr2[1], fArr2[2], fArr2[3]));
                                float[] fArr3 = veVar2.n;
                                if (fArr3[0] != StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                                    Q.setLocalScale(new Vector3(fArr3[0], fArr3[1], fArr3[2]).scaled(veVar2.p));
                                }
                                hdVar2.l.getArSceneView().getScene().addOnPeekTouchListener(new Scene.OnPeekTouchListener() { // from class: c.e.b.z5
                                    @Override // com.google.ar.sceneform.Scene.OnPeekTouchListener
                                    public final void onPeekTouch(HitTestResult hitTestResult, MotionEvent motionEvent) {
                                        hd.this.l.getTransformationSystem().onTouch(hitTestResult, motionEvent);
                                    }
                                });
                                c.b.a.a.a.S(f2, 0.5f, c.b.a.a.a.c(f2, 0.05f, simpleTransformableNode.getScaleController(), simpleTransformableNode)).setView(hdVar2.f4816g, R.layout.img_view).build().thenAccept(new Consumer() { // from class: c.e.b.w4
                                    @Override // java.util.function.Consumer
                                    public final void accept(Object obj) {
                                        hd hdVar3 = hd.this;
                                        String str = r;
                                        Node node2 = Q;
                                        ViewRenderable viewRenderable2 = (ViewRenderable) obj;
                                        Objects.requireNonNull(hdVar3);
                                        c.c.a.b.d(hdVar3.f4816g).k(str).B((ImageView) viewRenderable2.getView().findViewById(R.id.img_loader_view));
                                        viewRenderable2.setHorizontalAlignment(ViewRenderable.HorizontalAlignment.CENTER);
                                        viewRenderable2.setVerticalAlignment(ViewRenderable.VerticalAlignment.CENTER);
                                        node2.setRenderable(viewRenderable2);
                                    }
                                }).exceptionally(new Function() { // from class: c.e.b.p5
                                    @Override // java.util.function.Function
                                    public final Object apply(Object obj) {
                                        hd hdVar3 = hd.this;
                                        Node node2 = Q;
                                        Throwable th = (Throwable) obj;
                                        Objects.requireNonNull(hdVar3);
                                        Log.e("LoaderARContent", "Unable to load  youtube node");
                                        hdVar3.i.removeChild(node2);
                                        int i3 = hdVar3.r - 1;
                                        hdVar3.r = i3;
                                        if (i3 == 0) {
                                            hdVar3.k();
                                            return null;
                                        }
                                        return null;
                                    }
                                });
                                final Node node2 = new Node();
                                node2.setParent(Q);
                                node2.setLocalPosition(new Vector3(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 0.01f));
                                if (veVar2.f5355f.equals("PLAY_SOUND")) {
                                    mediaPlayer = hdVar2.m();
                                    mediaPlayer.setLooping(veVar2.i);
                                    hdVar2.g(veVar2, mediaPlayer);
                                    if (veVar2.f5357h) {
                                        hdVar2.J.put(mediaPlayer, Boolean.TRUE);
                                    }
                                } else {
                                    mediaPlayer = null;
                                }
                                final MediaPlayer mediaPlayer2 = mediaPlayer;
                                Q.setOnAccurateTapListner(new Node.OnAccurateTapListner() { // from class: c.e.b.a7
                                    @Override // com.google.ar.sceneform.Node.OnAccurateTapListner
                                    public final void onAccurateTap(HitTestResult hitTestResult, MotionEvent motionEvent) {
                                        hd hdVar3 = hd.this;
                                        ve veVar3 = veVar2;
                                        String str = a2;
                                        MediaPlayer mediaPlayer3 = mediaPlayer2;
                                        Node node3 = Q;
                                        Objects.requireNonNull(hdVar3);
                                        if (Objects.equals(veVar3.f5355f, "NO_ACTION")) {
                                            Intent intent = new Intent(hdVar3.f4815f, YoutubeView.class);
                                            intent.putExtra("youtubeID", str);
                                            hdVar3.f4815f.startActivity(intent);
                                            return;
                                        }
                                        hdVar3.b(veVar3, mediaPlayer3, node3);
                                    }
                                });
                                ViewRenderable.builder().setView(hdVar2.f4816g, R.layout.img_view).build().thenAccept(new Consumer() { // from class: c.e.b.r6
                                    @Override // java.util.function.Consumer
                                    public final void accept(Object obj) {
                                        hd hdVar3 = hd.this;
                                        Node node3 = node2;
                                        Node node4 = Q;
                                        ve veVar3 = veVar2;
                                        ViewRenderable viewRenderable2 = (ViewRenderable) obj;
                                        Objects.requireNonNull(hdVar3);
                                        ((ImageView) viewRenderable2.getView().findViewById(R.id.img_loader_view)).setImageResource(R.drawable.play_youtube);
                                        viewRenderable2.setHorizontalAlignment(ViewRenderable.HorizontalAlignment.CENTER);
                                        viewRenderable2.setVerticalAlignment(ViewRenderable.VerticalAlignment.CENTER);
                                        node3.setRenderable(viewRenderable2);
                                        c.b.a.a.a.C(0.2f, 0.2f, 0.2f, node3);
                                        hdVar3.L.put(node4, veVar3);
                                        node4.setEnabled(false);
                                        int i3 = hdVar3.r - 1;
                                        hdVar3.r = i3;
                                        if (i3 == 0) {
                                            hdVar3.k();
                                            return;
                                        }
                                        StringBuilder x = c.b.a.a.a.x("Your AR content is loading  (");
                                        c.b.a.a.a.D(hdVar3.s, hdVar3.r, x, "/");
                                        hdVar3.R.setText(c.b.a.a.a.s(x, hdVar3.s, ")"));
                                    }
                                }).exceptionally(new Function() { // from class: c.e.b.w6
                                    @Override // java.util.function.Function
                                    public final Object apply(Object obj) {
                                        hd hdVar3 = hd.this;
                                        Node node3 = node2;
                                        Throwable th = (Throwable) obj;
                                        Objects.requireNonNull(hdVar3);
                                        Log.e("LoaderARContent", "Unable to load  youtube button");
                                        hdVar3.i.removeChild(node3);
                                        int i3 = hdVar3.r - 1;
                                        hdVar3.r = i3;
                                        if (i3 == 0) {
                                            hdVar3.k();
                                            return null;
                                        }
                                        return null;
                                    }
                                });
                            }
                        });
                        break;
                    case 3:
                    case 5:
                        hdVar.f4816g.runOnUiThread(new Runnable() { // from class: c.e.b.t6
                            @Override // java.lang.Runnable
                            public final void run() {
                                final hd hdVar2 = hd.this;
                                final ve veVar2 = veVar;
                                Objects.requireNonNull(hdVar2);
                                float f2 = veVar2.m[1];
                                final SimpleTransformableNode simpleTransformableNode = new SimpleTransformableNode(hdVar2.l.getTransformationSystem());
                                simpleTransformableNode.setParent(hdVar2.i);
                                float[] fArr = veVar2.l;
                                final Node Q = c.b.a.a.a.Q(simpleTransformableNode, new Vector3(fArr[0], fArr[1], fArr[2]), simpleTransformableNode);
                                float[] fArr2 = veVar2.q;
                                Q.setLocalRotation(new Quaternion(-fArr2[0], -fArr2[1], fArr2[2], fArr2[3]));
                                float[] fArr3 = veVar2.n;
                                if (fArr3[0] != StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                                    Q.setLocalScale(new Vector3(fArr3[0], fArr3[1], fArr3[2]).scaled(veVar2.p));
                                }
                                hdVar2.l.getArSceneView().getScene().addOnPeekTouchListener(new Scene.OnPeekTouchListener() { // from class: c.e.b.c6
                                    @Override // com.google.ar.sceneform.Scene.OnPeekTouchListener
                                    public final void onPeekTouch(HitTestResult hitTestResult, MotionEvent motionEvent) {
                                        hd.this.l.getTransformationSystem().onTouch(hitTestResult, motionEvent);
                                    }
                                });
                                c.b.a.a.a.S(f2, 0.7f, c.b.a.a.a.c(f2, 0.07f, simpleTransformableNode.getScaleController(), simpleTransformableNode)).setView(hdVar2.f4816g, R.layout.image_view_threesixty).build().thenAccept(new Consumer() { // from class: c.e.b.h6
                                    @Override // java.util.function.Consumer
                                    public final void accept(Object obj) {
                                        hd hdVar3 = hd.this;
                                        Node node2 = Q;
                                        ve veVar3 = veVar2;
                                        ViewRenderable viewRenderable2 = (ViewRenderable) obj;
                                        Objects.requireNonNull(hdVar3);
                                        ((ImageView) viewRenderable2.getView().findViewById(R.id.threesixty_img)).setImageResource(R.drawable.button360);
                                        viewRenderable2.setHorizontalAlignment(ViewRenderable.HorizontalAlignment.CENTER);
                                        c.b.a.a.a.J(viewRenderable2, ViewRenderable.VerticalAlignment.CENTER, false, false);
                                        node2.setRenderable(viewRenderable2);
                                        hdVar3.L.put(node2, veVar3);
                                        node2.setEnabled(false);
                                        int i3 = hdVar3.r - 1;
                                        hdVar3.r = i3;
                                        if (i3 == 0) {
                                            hdVar3.k();
                                            return;
                                        }
                                        StringBuilder x = c.b.a.a.a.x("Your AR content is loading  (");
                                        c.b.a.a.a.D(hdVar3.s, hdVar3.r, x, "/");
                                        hdVar3.R.setText(c.b.a.a.a.s(x, hdVar3.s, ")"));
                                    }
                                }).exceptionally(new Function() { // from class: c.e.b.m6
                                    @Override // java.util.function.Function
                                    public final Object apply(Object obj) {
                                        hd hdVar3 = hd.this;
                                        SimpleTransformableNode simpleTransformableNode2 = simpleTransformableNode;
                                        Throwable th = (Throwable) obj;
                                        Objects.requireNonNull(hdVar3);
                                        Log.e("LoaderARContent", "Unable to load  360 node");
                                        hdVar3.i.removeChild(simpleTransformableNode2);
                                        int i3 = hdVar3.r - 1;
                                        hdVar3.r = i3;
                                        if (i3 == 0) {
                                            hdVar3.k();
                                            return null;
                                        }
                                        return null;
                                    }
                                });
                                Q.setOnAccurateTapListner(new Node.OnAccurateTapListner() { // from class: c.e.b.c5
                                    @Override // com.google.ar.sceneform.Node.OnAccurateTapListner
                                    public final void onAccurateTap(HitTestResult hitTestResult, MotionEvent motionEvent) {
                                        hd hdVar3 = hd.this;
                                        ve veVar3 = veVar2;
                                        Objects.requireNonNull(hdVar3);
                                        Intent intent = new Intent(hdVar3.f4815f, Player360Activity.class);
                                        intent.putExtra(ImagesContract.URL, veVar3.f5354e);
                                        hdVar3.f4815f.startActivity(intent);
                                    }
                                });
                                if (veVar2.k) {
                                    Intent intent = new Intent(hdVar2.f4815f, Player360Activity.class);
                                    intent.putExtra(ImagesContract.URL, veVar2.f5354e);
                                    hdVar2.f4815f.startActivity(intent);
                                }
                            }
                        });
                        break;
                    case 4:
                        hdVar.f4816g.runOnUiThread(new Runnable() { // from class: c.e.b.r4
                            @Override // java.lang.Runnable
                            public final void run() {
                                final MediaPlayer mediaPlayer;
                                final hd hdVar2 = hd.this;
                                final ve veVar2 = veVar;
                                Objects.requireNonNull(hdVar2);
                                final String[] split = veVar2.f5354e.split(",");
                                final int[] iArr = {0};
                                float f2 = veVar2.m[1];
                                final Node node2 = new Node();
                                final Node node3 = new Node();
                                final SimpleTransformableNode simpleTransformableNode = new SimpleTransformableNode(hdVar2.l.getTransformationSystem());
                                simpleTransformableNode.setParent(hdVar2.i);
                                float[] fArr = veVar2.l;
                                final Node Q = c.b.a.a.a.Q(simpleTransformableNode, new Vector3(fArr[0], fArr[1], fArr[2]), simpleTransformableNode);
                                float[] fArr2 = veVar2.q;
                                Q.setLocalRotation(new Quaternion(-fArr2[0], -fArr2[1], fArr2[2], fArr2[3]));
                                float[] fArr3 = veVar2.n;
                                if (fArr3[0] != StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                                    Q.setLocalScale(new Vector3(fArr3[0], fArr3[1], fArr3[2]).scaled(veVar2.p));
                                }
                                hdVar2.l.getArSceneView().getScene().addOnPeekTouchListener(new Scene.OnPeekTouchListener() { // from class: c.e.b.k4
                                    @Override // com.google.ar.sceneform.Scene.OnPeekTouchListener
                                    public final void onPeekTouch(HitTestResult hitTestResult, MotionEvent motionEvent) {
                                        hd.this.l.getTransformationSystem().onTouch(hitTestResult, motionEvent);
                                    }
                                });
                                simpleTransformableNode.getScaleController().setMinScale(0.04f);
                                simpleTransformableNode.getScaleController().setMaxScale(0.4f);
                                ViewRenderable.builder().setView(hdVar2.f4816g, R.layout.imageview_slideshow).build().thenAccept(new Consumer() { // from class: c.e.b.u6
                                    @Override // java.util.function.Consumer
                                    public final void accept(Object obj) {
                                        hd hdVar3 = hd.this;
                                        String[] strArr = split;
                                        Node node4 = node2;
                                        Node node5 = node3;
                                        Node node6 = Q;
                                        ve veVar3 = veVar2;
                                        int[] iArr2 = iArr;
                                        ViewRenderable viewRenderable2 = (ViewRenderable) obj;
                                        Objects.requireNonNull(hdVar3);
                                        ImageView imageView = (ImageView) viewRenderable2.getView().findViewById(R.id.img_loader_view);
                                        c.c.a.b.d(hdVar3.f4816g).k(strArr[0]).C(new md(hdVar3, node4, node5)).B(imageView);
                                        viewRenderable2.setHorizontalAlignment(ViewRenderable.HorizontalAlignment.CENTER);
                                        viewRenderable2.setVerticalAlignment(ViewRenderable.VerticalAlignment.CENTER);
                                        node6.setRenderable(viewRenderable2);
                                        if (veVar3.j) {
                                            Timer timer = new Timer();
                                            timer.schedule(new od(hdVar3, timer, iArr2, strArr, node4, node5, imageView), 0L, veVar3.O * 1000);
                                        } else if (strArr.length > 1) {
                                            viewRenderable2.setRenderPriority(4);
                                            hdVar3.q(node6, 1, iArr2, strArr, imageView, node4, node5);
                                            hdVar3.q(node6, -1, iArr2, strArr, imageView, node4, node5);
                                        }
                                        hdVar3.L.put(node6, veVar3);
                                        node6.setEnabled(false);
                                        int i3 = hdVar3.r - 1;
                                        hdVar3.r = i3;
                                        if (i3 == 0) {
                                            hdVar3.k();
                                            return;
                                        }
                                        StringBuilder x = c.b.a.a.a.x("Your AR content is loading  (");
                                        c.b.a.a.a.D(hdVar3.s, hdVar3.r, x, "/");
                                        hdVar3.R.setText(c.b.a.a.a.s(x, hdVar3.s, ")"));
                                    }
                                }).exceptionally(new Function() { // from class: c.e.b.u4
                                    @Override // java.util.function.Function
                                    public final Object apply(Object obj) {
                                        hd hdVar3 = hd.this;
                                        SimpleTransformableNode simpleTransformableNode2 = simpleTransformableNode;
                                        Throwable th = (Throwable) obj;
                                        Objects.requireNonNull(hdVar3);
                                        Log.e("LoaderARContent", "Unable to load  createImageSlideshowSceneform");
                                        hdVar3.i.removeChild(simpleTransformableNode2);
                                        int i3 = hdVar3.r - 1;
                                        hdVar3.r = i3;
                                        if (i3 == 0) {
                                            hdVar3.k();
                                            return null;
                                        }
                                        return null;
                                    }
                                });
                                if (veVar2.f5355f.equals("PLAY_SOUND")) {
                                    mediaPlayer = hdVar2.m();
                                    mediaPlayer.setLooping(veVar2.i);
                                    hdVar2.g(veVar2, mediaPlayer);
                                    if (veVar2.f5357h) {
                                        hdVar2.J.put(mediaPlayer, Boolean.TRUE);
                                    }
                                } else {
                                    mediaPlayer = null;
                                }
                                if (veVar2.f5355f.equals("PLAY_SOUND") && veVar2.f5357h) {
                                    hdVar2.g(veVar2, mediaPlayer);
                                }
                                Q.setOnAccurateTapListner(new Node.OnAccurateTapListner() { // from class: c.e.b.x4
                                    @Override // com.google.ar.sceneform.Node.OnAccurateTapListner
                                    public final void onAccurateTap(HitTestResult hitTestResult, MotionEvent motionEvent) {
                                        hd hdVar3 = hd.this;
                                        ve veVar3 = veVar2;
                                        MediaPlayer mediaPlayer2 = mediaPlayer;
                                        Objects.requireNonNull(hdVar3);
                                        if (Objects.equals(veVar3.f5355f, "GOTO_SCENE")) {
                                            hdVar3.C = veVar3.f5356g;
                                            hdVar3.n();
                                        } else if (Objects.equals(veVar3.f5355f, "PLAY_SOUND")) {
                                            if (mediaPlayer2.isPlaying()) {
                                                mediaPlayer2.pause();
                                            } else {
                                                mediaPlayer2.start();
                                            }
                                        }
                                    }
                                });
                            }
                        });
                        break;
                    case 6:
                        hdVar.f4816g.runOnUiThread(new Runnable() { // from class: c.e.b.l7
                            @Override // java.lang.Runnable
                            public final void run() {
                                MediaPlayer mediaPlayer;
                                final hd hdVar2 = hd.this;
                                final ve veVar2 = veVar;
                                Objects.requireNonNull(hdVar2);
                                float[] fArr = veVar2.m;
                                float max = Math.max(Math.max(fArr[0], fArr[1]), veVar2.m[2]);
                                SimpleTransformableNode simpleTransformableNode = new SimpleTransformableNode(hdVar2.l.getTransformationSystem());
                                simpleTransformableNode.setParent(hdVar2.i);
                                float[] fArr2 = veVar2.l;
                                simpleTransformableNode.setLocalPosition(new Vector3(fArr2[0], fArr2[1], fArr2[2]));
                                Node node2 = new Node();
                                simpleTransformableNode.setLocalRotation(Quaternion.eulerAngles(new Vector3(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD)));
                                node2.setParent(simpleTransformableNode);
                                float[] fArr3 = veVar2.q;
                                node2.setLocalRotation(Quaternion.multiply(Quaternion.eulerAngles(new Vector3(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 180.0f)), new Quaternion(fArr3[0], fArr3[1], fArr3[2], fArr3[3])));
                                final Node node3 = new Node();
                                node3.setParent(node2);
                                node3.setLocalRotation(Quaternion.eulerAngles(new Vector3(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 180.0f, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD)));
                                hdVar2.l.getArSceneView().getScene().addOnPeekTouchListener(new Scene.OnPeekTouchListener() { // from class: c.e.b.q4
                                    @Override // com.google.ar.sceneform.Scene.OnPeekTouchListener
                                    public final void onPeekTouch(HitTestResult hitTestResult, MotionEvent motionEvent) {
                                        hd.this.l.getTransformationSystem().onTouch(hitTestResult, motionEvent);
                                    }
                                });
                                if (veVar2.f5355f.equals("PLAY_SOUND")) {
                                    mediaPlayer = hdVar2.m();
                                    mediaPlayer.setLooping(veVar2.i);
                                    hdVar2.g(veVar2, mediaPlayer);
                                    if (veVar2.f5357h) {
                                        hdVar2.J.put(mediaPlayer, Boolean.TRUE);
                                    }
                                } else {
                                    mediaPlayer = null;
                                }
                                final MediaPlayer mediaPlayer2 = mediaPlayer;
                                if (veVar2.G.equals("LOCK_INTERACTIONS")) {
                                    simpleTransformableNode.setName("LOCK_INTERACTIONS");
                                    float[] fArr4 = veVar2.n;
                                    simpleTransformableNode.setLocalScale(new Vector3(fArr4[0] * 0.08f, fArr4[1] * 0.08f, fArr4[2] * 0.08f).scaled(veVar2.p));
                                } else {
                                    float[] fArr5 = veVar2.n;
                                    node2.setLocalScale(new Vector3(fArr5[0], fArr5[1], fArr5[2]).scaled(veVar2.p));
                                }
                                if (max == 1.0f) {
                                    simpleTransformableNode.getScaleController().setMinScale(0.02f);
                                    simpleTransformableNode.getScaleController().setMaxScale(0.14f);
                                } else {
                                    simpleTransformableNode.getScaleController().setMinScale(0.02f * max * 0.08f);
                                    simpleTransformableNode.getScaleController().setMaxScale(max * 0.14f * 0.08f);
                                }
                                String str = veVar2.f5354e;
                                File file = new File(hdVar2.f4815f.getCacheDir(), c.b.a.a.a.q("/assets/models/", str.substring(str.lastIndexOf(47) + 1)));
                                if (file.exists()) {
                                    hdVar2.t(file.getPath(), node3, veVar2, simpleTransformableNode, node2);
                                } else {
                                    String str2 = veVar2.f5354e;
                                    qd qdVar = new qd(hdVar2, node3, veVar2, simpleTransformableNode, node2);
                                    String[] split = str2.split("/");
                                    String str3 = split[split.length - 1];
                                    if (str3.toLowerCase().endsWith("glb")) {
                                        str3 = str3.replaceAll(".glb", "");
                                    }
                                    new c.e.b.p000if.k(str3, hdVar2.f4815f, qdVar).execute(str2);
                                }
                                node3.setOnAccurateTapListner(new Node.OnAccurateTapListner() { // from class: c.e.b.j7
                                    @Override // com.google.ar.sceneform.Node.OnAccurateTapListner
                                    public final void onAccurateTap(HitTestResult hitTestResult, MotionEvent motionEvent) {
                                        hd.this.b(veVar2, mediaPlayer2, node3);
                                    }
                                });
                            }
                        });
                        hdVar.I.put(Integer.valueOf(veVar.H), veVar);
                        break;
                    case 7:
                        hdVar.f4816g.runOnUiThread(new Runnable() { // from class: c.e.b.p6
                            @Override // java.lang.Runnable
                            public final void run() {
                                final hd hdVar2 = hd.this;
                                final ve veVar2 = veVar;
                                Objects.requireNonNull(hdVar2);
                                float f2 = veVar2.K.equalsIgnoreCase("fantasy") ? 0.52f : 0.67f;
                                f2 = (veVar2.K.equalsIgnoreCase("cursive") || veVar2.K.equalsIgnoreCase("serif")) ? 0.65f : 0.65f;
                                float f3 = 0.7f;
                                if (veVar2.K.equalsIgnoreCase("monospace") || veVar2.K.equalsIgnoreCase("serif")) {
                                    f3 = 0.82f;
                                } else if (veVar2.K.equalsIgnoreCase("tahoma")) {
                                    f3 = 0.75f;
                                }
                                float f4 = veVar2.m[1];
                                final SimpleTransformableNode simpleTransformableNode = new SimpleTransformableNode(hdVar2.l.getTransformationSystem());
                                simpleTransformableNode.setParent(hdVar2.i);
                                float[] fArr = veVar2.l;
                                final Node Q = c.b.a.a.a.Q(simpleTransformableNode, new Vector3(fArr[0], fArr[1], fArr[2]), simpleTransformableNode);
                                float[] fArr2 = veVar2.q;
                                Q.setLocalRotation(new Quaternion(-fArr2[0], -fArr2[1], fArr2[2], fArr2[3]));
                                float[] fArr3 = veVar2.n;
                                if (fArr3[0] != StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                                    Q.setLocalScale(new Vector3(f2 * fArr3[0], f3 * fArr3[1], fArr3[2] * 0.15f).scaled(veVar2.p));
                                } else {
                                    Q.setLocalScale(new Vector3(f2 * f4, f3 * f4, f4 * 0.15f));
                                }
                                hdVar2.l.getArSceneView().getScene().addOnPeekTouchListener(new Scene.OnPeekTouchListener() { // from class: c.e.b.j6
                                    @Override // com.google.ar.sceneform.Scene.OnPeekTouchListener
                                    public final void onPeekTouch(HitTestResult hitTestResult, MotionEvent motionEvent) {
                                        hd.this.l.getTransformationSystem().onTouch(hitTestResult, motionEvent);
                                    }
                                });
                                simpleTransformableNode.getScaleController().setMinScale(0.07f);
                                simpleTransformableNode.getScaleController().setMaxScale(0.8f);
                                if (veVar2.f5354e.length() == 0) {
                                    ViewRenderable.builder().setView(hdVar2.f4816g, R.layout.plain_button).build().thenAccept(new Consumer() { // from class: c.e.b.b6
                                        @Override // java.util.function.Consumer
                                        public final void accept(Object obj) {
                                            Typeface a2;
                                            hd hdVar3 = hd.this;
                                            ve veVar3 = veVar2;
                                            Node node2 = Q;
                                            ViewRenderable viewRenderable2 = (ViewRenderable) obj;
                                            Objects.requireNonNull(hdVar3);
                                            LinearLayout linearLayout = (LinearLayout) viewRenderable2.getView().findViewById(R.id.buttonViewContainers);
                                            TextView textView = (TextView) viewRenderable2.getView().findViewById(R.id.button_view_text);
                                            textView.setText(veVar3.r);
                                            if (veVar3.v.length() != 0) {
                                                textView.setTextColor(hdVar3.l(veVar3.v));
                                            }
                                            if (veVar3.x.length() != 0) {
                                                linearLayout.setBackgroundColor(Color.parseColor(veVar3.x.substring(0, 7)));
                                                if (veVar3.x.length() > 7) {
                                                    linearLayout.setAlpha(Integer.valueOf(veVar3.x.substring(7, 9), 16).intValue() / 255.0f);
                                                }
                                            }
                                            if (veVar3.K.length() != 0) {
                                                String lowerCase = veVar3.K.toLowerCase();
                                                lowerCase.hashCode();
                                                lowerCase.hashCode();
                                                char c2 = 65535;
                                                switch (lowerCase.hashCode()) {
                                                    case -1536685117:
                                                        if (lowerCase.equals("sans-serif")) {
                                                            c2 = 0;
                                                            break;
                                                        }
                                                        break;
                                                    case -1431958525:
                                                        if (lowerCase.equals("monospace")) {
                                                            c2 = 1;
                                                            break;
                                                        }
                                                        break;
                                                    case -1081737434:
                                                        if (lowerCase.equals("fantasy")) {
                                                            c2 = 2;
                                                            break;
                                                        }
                                                        break;
                                                    case -881195832:
                                                        if (lowerCase.equals("tahoma")) {
                                                            c2 = 3;
                                                            break;
                                                        }
                                                        break;
                                                    case -78847778:
                                                        if (lowerCase.equals("georgia")) {
                                                            c2 = 4;
                                                            break;
                                                        }
                                                        break;
                                                    case 109326717:
                                                        if (lowerCase.equals("serif")) {
                                                            c2 = 5;
                                                            break;
                                                        }
                                                        break;
                                                    case 1126973893:
                                                        if (lowerCase.equals("cursive")) {
                                                            c2 = 6;
                                                            break;
                                                        }
                                                        break;
                                                }
                                                switch (c2) {
                                                    case 0:
                                                        a2 = b.j.c.b.f.a(hdVar3.f4815f, R.font.sans_serif);
                                                        break;
                                                    case 1:
                                                        a2 = b.j.c.b.f.a(hdVar3.f4815f, R.font.monospace);
                                                        break;
                                                    case 2:
                                                        a2 = b.j.c.b.f.a(hdVar3.f4815f, R.font.fantasy);
                                                        break;
                                                    case 3:
                                                        a2 = b.j.c.b.f.a(hdVar3.f4815f, R.font.tahoma);
                                                        break;
                                                    case 4:
                                                        a2 = b.j.c.b.f.a(hdVar3.f4815f, R.font.georgia);
                                                        break;
                                                    case 5:
                                                        a2 = Typeface.create(veVar3.K, 0);
                                                        break;
                                                    case 6:
                                                        a2 = b.j.c.b.f.a(hdVar3.f4815f, R.font.cursive);
                                                        break;
                                                    default:
                                                        a2 = null;
                                                        break;
                                                }
                                                if (a2 != null) {
                                                    textView.setTypeface(a2);
                                                }
                                            }
                                            viewRenderable2.setHorizontalAlignment(ViewRenderable.HorizontalAlignment.CENTER);
                                            c.b.a.a.a.J(viewRenderable2, ViewRenderable.VerticalAlignment.CENTER, false, false);
                                            node2.setRenderable(viewRenderable2);
                                            hdVar3.L.put(node2, veVar3);
                                            node2.setEnabled(false);
                                            int i3 = hdVar3.r - 1;
                                            hdVar3.r = i3;
                                            if (i3 == 0) {
                                                hdVar3.k();
                                                return;
                                            }
                                            StringBuilder x = c.b.a.a.a.x("Your AR content is loading  (");
                                            c.b.a.a.a.D(hdVar3.s, hdVar3.r, x, "/");
                                            hdVar3.R.setText(c.b.a.a.a.s(x, hdVar3.s, ")"));
                                        }
                                    }).exceptionally(new Function() { // from class: c.e.b.c7
                                        @Override // java.util.function.Function
                                        public final Object apply(Object obj) {
                                            hd hdVar3 = hd.this;
                                            SimpleTransformableNode simpleTransformableNode2 = simpleTransformableNode;
                                            Objects.requireNonNull(hdVar3);
                                            StringBuilder sb = new StringBuilder();
                                            sb.append("Unable to load button ");
                                            c.b.a.a.a.N((Throwable) obj, sb, "LoaderARContent");
                                            hdVar3.i.removeChild(simpleTransformableNode2);
                                            int i3 = hdVar3.r - 1;
                                            hdVar3.r = i3;
                                            if (i3 == 0) {
                                                hdVar3.k();
                                                return null;
                                            }
                                            return null;
                                        }
                                    });
                                } else {
                                    ViewRenderable.builder().setView(hdVar2.f4816g, R.layout.imagebutton).build().thenAccept(new Consumer() { // from class: c.e.b.f6
                                        @Override // java.util.function.Consumer
                                        public final void accept(Object obj) {
                                            hd hdVar3 = hd.this;
                                            ve veVar3 = veVar2;
                                            Node node2 = Q;
                                            ViewRenderable viewRenderable2 = (ViewRenderable) obj;
                                            Objects.requireNonNull(hdVar3);
                                            Log.d("LoaderARContent", "Building custom image button ViewRenderable");
                                            c.c.a.b.e(hdVar3.f4815f).k(veVar3.f5354e).B((ImageView) viewRenderable2.getView().findViewById(R.id.button_view));
                                            viewRenderable2.setHorizontalAlignment(ViewRenderable.HorizontalAlignment.CENTER);
                                            c.b.a.a.a.J(viewRenderable2, ViewRenderable.VerticalAlignment.CENTER, false, false);
                                            node2.setRenderable(viewRenderable2);
                                            hdVar3.L.put(node2, veVar3);
                                            node2.setEnabled(false);
                                            int i3 = hdVar3.r - 1;
                                            hdVar3.r = i3;
                                            if (i3 == 0) {
                                                hdVar3.k();
                                                return;
                                            }
                                            StringBuilder x = c.b.a.a.a.x("Your AR content is loading  (");
                                            c.b.a.a.a.D(hdVar3.s, hdVar3.r, x, "/");
                                            hdVar3.R.setText(c.b.a.a.a.s(x, hdVar3.s, ")"));
                                        }
                                    }).exceptionally(new Function() { // from class: c.e.b.e7
                                        @Override // java.util.function.Function
                                        public final Object apply(Object obj) {
                                            hd hdVar3 = hd.this;
                                            SimpleTransformableNode simpleTransformableNode2 = simpleTransformableNode;
                                            Throwable th = (Throwable) obj;
                                            Objects.requireNonNull(hdVar3);
                                            Log.e("LoaderARContent", "Unable to load  custom button");
                                            hdVar3.i.removeChild(simpleTransformableNode2);
                                            int i3 = hdVar3.r - 1;
                                            hdVar3.r = i3;
                                            if (i3 == 0) {
                                                hdVar3.k();
                                                return null;
                                            }
                                            return null;
                                        }
                                    });
                                }
                                Q.setOnAccurateTapListner(new Node.OnAccurateTapListner() { // from class: c.e.b.h5
                                    @Override // com.google.ar.sceneform.Node.OnAccurateTapListner
                                    public final void onAccurateTap(HitTestResult hitTestResult, MotionEvent motionEvent) {
                                        String str;
                                        ObjectAnimator objectAnimator;
                                        hd hdVar3 = hd.this;
                                        ve veVar3 = veVar2;
                                        Objects.requireNonNull(hdVar3);
                                        if (Objects.equals(veVar3.S, "2.0.0")) {
                                            str = veVar3.f5355f;
                                        } else {
                                            str = veVar3.t;
                                        }
                                        String str2 = Objects.equals(veVar3.s, "") ? veVar3.f5356g : veVar3.s;
                                        str.hashCode();
                                        char c2 = 65535;
                                        switch (str.hashCode()) {
                                            case -1858525552:
                                                if (str.equals("GOTO_SCENE")) {
                                                    c2 = 0;
                                                    break;
                                                }
                                                break;
                                            case -1235449741:
                                                if (str.equals("GOTO_URL")) {
                                                    c2 = 1;
                                                    break;
                                                }
                                                break;
                                            case -582612232:
                                                if (str.equals("DIAL_NUMBER")) {
                                                    c2 = 2;
                                                    break;
                                                }
                                                break;
                                            case -211278749:
                                                if (str.equals("TRIGGER_STOP_CONTENT")) {
                                                    c2 = 3;
                                                    break;
                                                }
                                                break;
                                            case 49:
                                                if (str.equals("1")) {
                                                    c2 = 4;
                                                    break;
                                                }
                                                break;
                                            case 50:
                                                if (str.equals("2")) {
                                                    c2 = 5;
                                                    break;
                                                }
                                                break;
                                            case 52:
                                                if (str.equals("4")) {
                                                    c2 = 6;
                                                    break;
                                                }
                                                break;
                                            case 53:
                                                if (str.equals("5")) {
                                                    c2 = 7;
                                                    break;
                                                }
                                                break;
                                            case 261182869:
                                                if (str.equals("TRIGGER_PLAY_CONTENT")) {
                                                    c2 = '\b';
                                                    break;
                                                }
                                                break;
                                            case 1125406319:
                                                if (str.equals("COMPOSE_EMAIL")) {
                                                    c2 = '\t';
                                                    break;
                                                }
                                                break;
                                            case 1464932140:
                                                if (str.equals("TRIGGER_PLAY_PAUSE_CONTENT")) {
                                                    c2 = '\n';
                                                    break;
                                                }
                                                break;
                                            case 1728936361:
                                                if (str.equals("TRIGGER_REPLACE_TEXTURE")) {
                                                    c2 = 11;
                                                    break;
                                                }
                                                break;
                                        }
                                        switch (c2) {
                                            case 0:
                                            case 7:
                                                hdVar3.C = str2;
                                                hdVar3.n();
                                                return;
                                            case 1:
                                            case 5:
                                                hdVar3.o(str2);
                                                return;
                                            case 2:
                                            case 4:
                                                hdVar3.j(str2);
                                                return;
                                            case 3:
                                                if (hdVar3.H.get(Integer.valueOf(veVar3.P)) == null || (objectAnimator = hdVar3.E.get(Integer.valueOf(veVar3.P))) == null) {
                                                    return;
                                                }
                                                if (objectAnimator.isRunning() || objectAnimator.isPaused()) {
                                                    objectAnimator.pause();
                                                    return;
                                                }
                                                return;
                                            case 6:
                                            case '\t':
                                                hdVar3.x(str2);
                                                return;
                                            case '\b':
                                                if (hdVar3.H.get(Integer.valueOf(veVar3.P)) == null) {
                                                    return;
                                                }
                                                ObjectAnimator objectAnimator2 = hdVar3.E.get(Integer.valueOf(veVar3.P));
                                                if (objectAnimator2 != null) {
                                                    objectAnimator2.pause();
                                                    objectAnimator2.removeAllUpdateListeners();
                                                    objectAnimator2.removeAllListeners();
                                                    objectAnimator2.end();
                                                    objectAnimator2.cancel();
                                                }
                                                Node node2 = hdVar3.H.get(Integer.valueOf(veVar3.P));
                                                Objects.requireNonNull(node2);
                                                hdVar3.f(node2.getRenderableInstance(), veVar3.R, veVar3.P, veVar3.Q, true, "trigger");
                                                return;
                                            case '\n':
                                                if (hdVar3.H.get(Integer.valueOf(veVar3.P)) == null) {
                                                    return;
                                                }
                                                ObjectAnimator objectAnimator3 = hdVar3.E.get(Integer.valueOf(veVar3.P));
                                                if (objectAnimator3 != null) {
                                                    String[] strArr = hdVar3.F.get(Integer.valueOf(veVar3.P));
                                                    Objects.requireNonNull(strArr);
                                                    if (strArr[0].toLowerCase().contains(veVar3.Q.toLowerCase())) {
                                                        if (!objectAnimator3.isPaused() && objectAnimator3.isRunning()) {
                                                            objectAnimator3.pause();
                                                            return;
                                                        } else {
                                                            objectAnimator3.start();
                                                            return;
                                                        }
                                                    }
                                                    objectAnimator3.pause();
                                                    objectAnimator3.removeAllUpdateListeners();
                                                    objectAnimator3.removeAllListeners();
                                                    objectAnimator3.end();
                                                    objectAnimator3.cancel();
                                                    Node node3 = hdVar3.H.get(Integer.valueOf(veVar3.P));
                                                    Objects.requireNonNull(node3);
                                                    hdVar3.f(node3.getRenderableInstance(), veVar3.R, veVar3.P, veVar3.Q, true, "trigger");
                                                    return;
                                                }
                                                Node node4 = hdVar3.H.get(Integer.valueOf(veVar3.P));
                                                Objects.requireNonNull(node4);
                                                hdVar3.f(node4.getRenderableInstance(), veVar3.R, veVar3.P, veVar3.Q, true, "trigger");
                                                return;
                                            case 11:
                                                if (hdVar3.H.get(Integer.valueOf(veVar3.P)) != null) {
                                                    hdVar3.p(hdVar3.M.get(Integer.valueOf(veVar3.P)), hdVar3.H.get(Integer.valueOf(veVar3.P)), (String) hdVar3.O.get(Integer.valueOf(veVar3.P)).get(Integer.valueOf(veVar3.f5353d)), veVar3.P);
                                                    return;
                                                }
                                                return;
                                            default:
                                                return;
                                        }
                                    }
                                });
                            }
                        });
                        break;
                    case 8:
                        hdVar.f4816g.runOnUiThread(new Runnable() { // from class: c.e.b.l6
                            @Override // java.lang.Runnable
                            public final void run() {
                                final hd hdVar2 = hd.this;
                                final ve veVar2 = veVar;
                                Objects.requireNonNull(hdVar2);
                                float f2 = veVar2.m[1];
                                final SimpleTransformableNode simpleTransformableNode = new SimpleTransformableNode(hdVar2.l.getTransformationSystem());
                                simpleTransformableNode.setParent(hdVar2.i);
                                float[] fArr = veVar2.l;
                                final Node Q = c.b.a.a.a.Q(simpleTransformableNode, new Vector3(fArr[0], fArr[1], fArr[2]), simpleTransformableNode);
                                float[] fArr2 = veVar2.q;
                                Q.setLocalRotation(new Quaternion(-fArr2[0], -fArr2[1], fArr2[2], fArr2[3]));
                                float[] fArr3 = veVar2.n;
                                if (fArr3[0] != StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                                    Q.setLocalScale(new Vector3(fArr3[0], fArr3[1], fArr3[2]).scaled(veVar2.p));
                                }
                                hdVar2.l.getArSceneView().getScene().addOnPeekTouchListener(new Scene.OnPeekTouchListener() { // from class: c.e.b.o5
                                    @Override // com.google.ar.sceneform.Scene.OnPeekTouchListener
                                    public final void onPeekTouch(HitTestResult hitTestResult, MotionEvent motionEvent) {
                                        hd.this.l.getTransformationSystem().onTouch(hitTestResult, motionEvent);
                                    }
                                });
                                c.b.a.a.a.c(f2, 0.02f, simpleTransformableNode.getScaleController(), simpleTransformableNode).setMaxScale(f2 * 0.16f);
                                final MediaPlayer m = hdVar2.m();
                                hdVar2.g(veVar2, m);
                                m.setOnCompletionListener(new MediaPlayer.OnCompletionListener() { // from class: c.e.b.f5
                                    @Override // android.media.MediaPlayer.OnCompletionListener
                                    public final void onCompletion(MediaPlayer mediaPlayer) {
                                        hd hdVar3 = hd.this;
                                        ve veVar3 = veVar2;
                                        Objects.requireNonNull(hdVar3);
                                        if (Objects.equals(veVar3.f5355f, "GOTO_SCENE_AFTER_CONTENT_END")) {
                                            hdVar3.C = veVar3.f5356g;
                                            hdVar3.n();
                                        }
                                    }
                                });
                                m.setAudioStreamType(3);
                                try {
                                    m.setScreenOnWhilePlaying(true);
                                    m.setDataSource(veVar2.f5354e);
                                    if (veVar2.o) {
                                        m.setLooping(true);
                                    }
                                    m.prepareAsync();
                                    m.setOnPreparedListener(new MediaPlayer.OnPreparedListener() { // from class: c.e.b.e5
                                        @Override // android.media.MediaPlayer.OnPreparedListener
                                        public final void onPrepared(MediaPlayer mediaPlayer) {
                                            final hd hdVar3 = hd.this;
                                            final ve veVar3 = veVar2;
                                            final Node node2 = Q;
                                            final MediaPlayer mediaPlayer2 = m;
                                            final SimpleTransformableNode simpleTransformableNode2 = simpleTransformableNode;
                                            Objects.requireNonNull(hdVar3);
                                            ViewRenderable.builder().setView(hdVar3.f4816g, R.layout.img_view).build().thenAccept(new Consumer() { // from class: c.e.b.u5
                                                @Override // java.util.function.Consumer
                                                public final void accept(Object obj) {
                                                    hd hdVar4 = hd.this;
                                                    ve veVar4 = veVar3;
                                                    Node node3 = node2;
                                                    MediaPlayer mediaPlayer3 = mediaPlayer2;
                                                    ViewRenderable viewRenderable2 = (ViewRenderable) obj;
                                                    Objects.requireNonNull(hdVar4);
                                                    ImageView imageView = (ImageView) viewRenderable2.getView().findViewById(R.id.img_loader_view);
                                                    hdVar4.P = imageView;
                                                    imageView.setImageResource(R.drawable.audio);
                                                    hdVar4.P.setColorFilter(hdVar4.l(veVar4.N), PorterDuff.Mode.MULTIPLY);
                                                    if (veVar4.M.length() != 0) {
                                                        hdVar4.P.setBackgroundColor(Color.parseColor(veVar4.M.substring(0, 7)));
                                                        if (veVar4.M.length() > 7) {
                                                            hdVar4.P.setAlpha(Integer.valueOf(veVar4.M.substring(7, 9), 16).intValue() / 255.0f);
                                                        }
                                                    }
                                                    int i3 = (int) ((hdVar4.f4815f.getResources().getDisplayMetrics().density * 36.0f) + 0.5f);
                                                    hdVar4.P.setPadding(i3, i3, i3, i3);
                                                    viewRenderable2.setHorizontalAlignment(ViewRenderable.HorizontalAlignment.CENTER);
                                                    viewRenderable2.setVerticalAlignment(ViewRenderable.VerticalAlignment.CENTER);
                                                    node3.setRenderable(viewRenderable2);
                                                    hdVar4.L.put(node3, veVar4);
                                                    node3.setEnabled(false);
                                                    if (veVar4.j) {
                                                        hdVar4.J.put(mediaPlayer3, Boolean.TRUE);
                                                        hdVar4.P.setImageResource(R.drawable.audio);
                                                    } else {
                                                        hdVar4.P.setImageResource(R.drawable.audio_mute);
                                                    }
                                                    int i4 = hdVar4.r - 1;
                                                    hdVar4.r = i4;
                                                    if (i4 == 0) {
                                                        hdVar4.k();
                                                        return;
                                                    }
                                                    StringBuilder x = c.b.a.a.a.x("Your AR content is loading  (");
                                                    c.b.a.a.a.D(hdVar4.s, hdVar4.r, x, "/");
                                                    hdVar4.R.setText(c.b.a.a.a.s(x, hdVar4.s, ")"));
                                                }
                                            }).exceptionally(new Function() { // from class: c.e.b.p4
                                                @Override // java.util.function.Function
                                                public final Object apply(Object obj) {
                                                    hd hdVar4 = hd.this;
                                                    SimpleTransformableNode simpleTransformableNode3 = simpleTransformableNode2;
                                                    Objects.requireNonNull(hdVar4);
                                                    StringBuilder sb = new StringBuilder();
                                                    sb.append("Unable to load  audio ");
                                                    c.b.a.a.a.N((Throwable) obj, sb, "LoaderARContent");
                                                    hdVar4.i.removeChild(simpleTransformableNode3);
                                                    int i3 = hdVar4.r - 1;
                                                    hdVar4.r = i3;
                                                    if (i3 == 0) {
                                                        hdVar4.k();
                                                        return null;
                                                    }
                                                    return null;
                                                }
                                            });
                                        }
                                    });
                                } catch (IOException e2) {
                                    e2.printStackTrace();
                                }
                                Q.setOnAccurateTapListner(new Node.OnAccurateTapListner() { // from class: c.e.b.m7
                                    @Override // com.google.ar.sceneform.Node.OnAccurateTapListner
                                    public final void onAccurateTap(HitTestResult hitTestResult, MotionEvent motionEvent) {
                                        hd hdVar3 = hd.this;
                                        ve veVar3 = veVar2;
                                        MediaPlayer mediaPlayer = m;
                                        Node node2 = Q;
                                        Objects.requireNonNull(hdVar3);
                                        ImageView imageView = (ImageView) ((ViewRenderable) node2.getRenderable()).getView().findViewById(R.id.img_loader_view);
                                        if (!Objects.equals(veVar3.f5355f, "PLAY_PAUSE_CONTENT") && !Objects.equals(veVar3.f5355f, "GOTO_SCENE_AFTER_CONTENT_END")) {
                                            if (Objects.equals(veVar3.f5355f, "GOTO_SCENE")) {
                                                hdVar3.C = veVar3.f5356g;
                                                hdVar3.n();
                                            }
                                        } else if (mediaPlayer.isPlaying()) {
                                            mediaPlayer.pause();
                                            imageView.setImageResource(R.drawable.audio_mute);
                                        } else {
                                            mediaPlayer.start();
                                            imageView.setImageResource(R.drawable.audio);
                                        }
                                    }
                                });
                            }
                        });
                        break;
                    case 9:
                        hdVar.f4816g.runOnUiThread(new Runnable() { // from class: c.e.b.n7
                            @Override // java.lang.Runnable
                            public final void run() {
                                final MediaPlayer mediaPlayer;
                                final hd hdVar2 = hd.this;
                                final ve veVar2 = veVar;
                                Objects.requireNonNull(hdVar2);
                                float f2 = veVar2.m[1];
                                final SimpleTransformableNode simpleTransformableNode = new SimpleTransformableNode(hdVar2.l.getTransformationSystem());
                                simpleTransformableNode.setParent(hdVar2.i);
                                float[] fArr = veVar2.l;
                                final Node Q = c.b.a.a.a.Q(simpleTransformableNode, new Vector3(fArr[0], fArr[1], fArr[2]), simpleTransformableNode);
                                float[] fArr2 = veVar2.q;
                                Q.setLocalRotation(new Quaternion(-fArr2[0], -fArr2[1], fArr2[2], fArr2[3]));
                                float[] fArr3 = veVar2.n;
                                if (fArr3[0] != StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                                    Q.setLocalScale(new Vector3(fArr3[0], fArr3[1], fArr3[2]).scaled(veVar2.p));
                                }
                                hdVar2.l.getArSceneView().getScene().addOnPeekTouchListener(new Scene.OnPeekTouchListener() { // from class: c.e.b.d7
                                    @Override // com.google.ar.sceneform.Scene.OnPeekTouchListener
                                    public final void onPeekTouch(HitTestResult hitTestResult, MotionEvent motionEvent) {
                                        hd.this.l.getTransformationSystem().onTouch(hitTestResult, motionEvent);
                                    }
                                });
                                c.b.a.a.a.c(f2, 0.036f, simpleTransformableNode.getScaleController(), simpleTransformableNode).setMaxScale(f2 * 0.36f);
                                if (veVar2.f5355f.equals("PLAY_SOUND")) {
                                    mediaPlayer = hdVar2.m();
                                    mediaPlayer.setLooping(veVar2.i);
                                    hdVar2.g(veVar2, mediaPlayer);
                                    if (veVar2.f5357h) {
                                        hdVar2.J.put(mediaPlayer, Boolean.TRUE);
                                    }
                                } else {
                                    mediaPlayer = null;
                                }
                                Q.setOnAccurateTapListner(new Node.OnAccurateTapListner() { // from class: c.e.b.v5
                                    @Override // com.google.ar.sceneform.Node.OnAccurateTapListner
                                    public final void onAccurateTap(HitTestResult hitTestResult, MotionEvent motionEvent) {
                                        hd hdVar3 = hd.this;
                                        ve veVar3 = veVar2;
                                        MediaPlayer mediaPlayer2 = mediaPlayer;
                                        Objects.requireNonNull(hdVar3);
                                        if (Objects.equals(veVar3.f5355f, "GOTO_SCENE")) {
                                            hdVar3.C = veVar3.f5356g;
                                            hdVar3.n();
                                        } else if (Objects.equals(veVar3.f5355f, "PLAY_SOUND")) {
                                            if (mediaPlayer2.isPlaying()) {
                                                mediaPlayer2.pause();
                                            } else {
                                                mediaPlayer2.start();
                                            }
                                        }
                                    }
                                });
                                ViewRenderable.builder().setView(hdVar2.f4816g, R.layout.img_view).build().thenAccept(new Consumer() { // from class: c.e.b.h7
                                    @Override // java.util.function.Consumer
                                    public final void accept(Object obj) {
                                        hd hdVar3 = hd.this;
                                        ve veVar3 = veVar2;
                                        Node node2 = Q;
                                        ViewRenderable viewRenderable2 = (ViewRenderable) obj;
                                        Objects.requireNonNull(hdVar3);
                                        c.c.a.b.d(hdVar3.f4816g).k(veVar3.f5354e).B((ImageView) viewRenderable2.getView().findViewById(R.id.img_loader_view));
                                        viewRenderable2.setHorizontalAlignment(ViewRenderable.HorizontalAlignment.CENTER);
                                        viewRenderable2.setVerticalAlignment(ViewRenderable.VerticalAlignment.CENTER);
                                        node2.setRenderable(viewRenderable2);
                                        hdVar3.L.put(node2, veVar3);
                                        node2.setEnabled(false);
                                        int i3 = hdVar3.r - 1;
                                        hdVar3.r = i3;
                                        if (i3 == 0) {
                                            hdVar3.k();
                                            return;
                                        }
                                        StringBuilder x = c.b.a.a.a.x("Your AR content is loading  (");
                                        c.b.a.a.a.D(hdVar3.s, hdVar3.r, x, "/");
                                        hdVar3.R.setText(c.b.a.a.a.s(x, hdVar3.s, ")"));
                                    }
                                }).exceptionally(new Function() { // from class: c.e.b.i7
                                    @Override // java.util.function.Function
                                    public final Object apply(Object obj) {
                                        hd hdVar3 = hd.this;
                                        SimpleTransformableNode simpleTransformableNode2 = simpleTransformableNode;
                                        Throwable th = (Throwable) obj;
                                        Objects.requireNonNull(hdVar3);
                                        Log.e("LoaderARContent", "Unable to load  createGIFSceneform");
                                        hdVar3.i.removeChild(simpleTransformableNode2);
                                        int i3 = hdVar3.r - 1;
                                        hdVar3.r = i3;
                                        if (i3 == 0) {
                                            hdVar3.k();
                                            return null;
                                        }
                                        return null;
                                    }
                                });
                            }
                        });
                        break;
                    case 10:
                        hdVar.f4816g.runOnUiThread(new Runnable() { // from class: c.e.b.t5
                            @Override // java.lang.Runnable
                            public final void run() {
                                final MediaPlayer mediaPlayer;
                                final hd hdVar2 = hd.this;
                                final ve veVar2 = veVar;
                                Objects.requireNonNull(hdVar2);
                                float f2 = veVar2.K.equals("fantasy") ? 1.2f : 1.05f;
                                float f3 = veVar2.K.equals("monospace") ? 1.0f : 0.8f;
                                float f4 = veVar2.m[1];
                                final SimpleTransformableNode simpleTransformableNode = new SimpleTransformableNode(hdVar2.l.getTransformationSystem());
                                simpleTransformableNode.setParent(hdVar2.i);
                                float[] fArr = veVar2.l;
                                final Node Q = c.b.a.a.a.Q(simpleTransformableNode, new Vector3(fArr[0], fArr[1], fArr[2]), simpleTransformableNode);
                                float[] fArr2 = veVar2.q;
                                Q.setLocalRotation(new Quaternion(-fArr2[0], -fArr2[1], fArr2[2], fArr2[3]));
                                float[] fArr3 = veVar2.n;
                                if (fArr3[0] != StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                                    Q.setLocalScale(new Vector3(f2 * fArr3[0], f3 * fArr3[1], fArr3[2] * 0.15f).scaled(veVar2.p));
                                } else {
                                    Q.setLocalScale(new Vector3(f2 * f4, f3 * f4, f4 * 0.15f));
                                }
                                hdVar2.l.getArSceneView().getScene().addOnPeekTouchListener(new Scene.OnPeekTouchListener() { // from class: c.e.b.g7
                                    @Override // com.google.ar.sceneform.Scene.OnPeekTouchListener
                                    public final void onPeekTouch(HitTestResult hitTestResult, MotionEvent motionEvent) {
                                        hd.this.l.getTransformationSystem().onTouch(hitTestResult, motionEvent);
                                    }
                                });
                                if (veVar2.f5355f.equals("PLAY_SOUND")) {
                                    mediaPlayer = hdVar2.m();
                                    mediaPlayer.setLooping(veVar2.i);
                                    hdVar2.g(veVar2, mediaPlayer);
                                    if (veVar2.f5357h) {
                                        hdVar2.J.put(mediaPlayer, Boolean.TRUE);
                                    }
                                } else {
                                    mediaPlayer = null;
                                }
                                simpleTransformableNode.getScaleController().setMinScale(0.07f);
                                simpleTransformableNode.getScaleController().setMaxScale(0.7f);
                                ViewRenderable.builder().setView(hdVar2.f4816g, R.layout.text).build().thenAccept(new Consumer() { // from class: c.e.b.x5
                                    @Override // java.util.function.Consumer
                                    public final void accept(Object obj) {
                                        Typeface a2;
                                        hd hdVar3 = hd.this;
                                        ve veVar3 = veVar2;
                                        Node node2 = Q;
                                        ViewRenderable viewRenderable2 = (ViewRenderable) obj;
                                        Objects.requireNonNull(hdVar3);
                                        LinearLayout linearLayout = (LinearLayout) viewRenderable2.getView().findViewById(R.id.textViewContainers);
                                        TextView textView = (TextView) viewRenderable2.getView().findViewById(R.id.text_views);
                                        String[] split = veVar3.L.split(" ");
                                        StringBuilder sb = new StringBuilder();
                                        int i3 = 0;
                                        while (i3 < split.length) {
                                            sb.append(split[i3]);
                                            sb.append(" ");
                                            i3++;
                                            if (i3 % 5 == 0) {
                                                sb.append("\n");
                                            }
                                        }
                                        textView.setText(sb.toString().trim());
                                        if (veVar3.v.length() != 0) {
                                            textView.setTextColor(hdVar3.l(veVar3.v));
                                        }
                                        if (veVar3.J.length() != 0) {
                                            linearLayout.setBackgroundColor(Color.parseColor(veVar3.J.substring(0, 7)));
                                            if (veVar3.J.length() > 7) {
                                                linearLayout.setAlpha(Integer.valueOf("FF", 16).intValue() / 255.0f);
                                            }
                                        }
                                        if (veVar3.K.length() != 0) {
                                            String lowerCase = veVar3.K.toLowerCase();
                                            lowerCase.hashCode();
                                            lowerCase.hashCode();
                                            char c2 = 65535;
                                            switch (lowerCase.hashCode()) {
                                                case -1536685117:
                                                    if (lowerCase.equals("sans-serif")) {
                                                        c2 = 0;
                                                        break;
                                                    }
                                                    break;
                                                case -1431958525:
                                                    if (lowerCase.equals("monospace")) {
                                                        c2 = 1;
                                                        break;
                                                    }
                                                    break;
                                                case -1081737434:
                                                    if (lowerCase.equals("fantasy")) {
                                                        c2 = 2;
                                                        break;
                                                    }
                                                    break;
                                                case -881195832:
                                                    if (lowerCase.equals("tahoma")) {
                                                        c2 = 3;
                                                        break;
                                                    }
                                                    break;
                                                case -78847778:
                                                    if (lowerCase.equals("georgia")) {
                                                        c2 = 4;
                                                        break;
                                                    }
                                                    break;
                                                case 109326717:
                                                    if (lowerCase.equals("serif")) {
                                                        c2 = 5;
                                                        break;
                                                    }
                                                    break;
                                                case 1126973893:
                                                    if (lowerCase.equals("cursive")) {
                                                        c2 = 6;
                                                        break;
                                                    }
                                                    break;
                                            }
                                            switch (c2) {
                                                case 0:
                                                    a2 = b.j.c.b.f.a(hdVar3.f4815f, R.font.sans_serif);
                                                    break;
                                                case 1:
                                                    a2 = b.j.c.b.f.a(hdVar3.f4815f, R.font.monospace);
                                                    break;
                                                case 2:
                                                    a2 = b.j.c.b.f.a(hdVar3.f4815f, R.font.fantasy);
                                                    break;
                                                case 3:
                                                    a2 = b.j.c.b.f.a(hdVar3.f4815f, R.font.tahoma);
                                                    break;
                                                case 4:
                                                    a2 = b.j.c.b.f.a(hdVar3.f4815f, R.font.georgia);
                                                    break;
                                                case 5:
                                                    a2 = Typeface.create(veVar3.K, 0);
                                                    break;
                                                case 6:
                                                    a2 = b.j.c.b.f.a(hdVar3.f4815f, R.font.cursive);
                                                    break;
                                                default:
                                                    a2 = null;
                                                    break;
                                            }
                                            if (a2 != null) {
                                                textView.setTypeface(a2);
                                            }
                                        }
                                        viewRenderable2.setHorizontalAlignment(ViewRenderable.HorizontalAlignment.CENTER);
                                        c.b.a.a.a.J(viewRenderable2, ViewRenderable.VerticalAlignment.CENTER, false, false);
                                        node2.setRenderable(viewRenderable2);
                                        hdVar3.L.put(node2, veVar3);
                                        node2.setEnabled(false);
                                        int i4 = hdVar3.r - 1;
                                        hdVar3.r = i4;
                                        if (i4 == 0) {
                                            hdVar3.k();
                                            return;
                                        }
                                        StringBuilder x = c.b.a.a.a.x("Your AR content is loading  (");
                                        c.b.a.a.a.D(hdVar3.s, hdVar3.r, x, "/");
                                        hdVar3.R.setText(c.b.a.a.a.s(x, hdVar3.s, ")"));
                                    }
                                }).exceptionally(new Function() { // from class: c.e.b.i6
                                    @Override // java.util.function.Function
                                    public final Object apply(Object obj) {
                                        hd hdVar3 = hd.this;
                                        SimpleTransformableNode simpleTransformableNode2 = simpleTransformableNode;
                                        Objects.requireNonNull(hdVar3);
                                        StringBuilder sb = new StringBuilder();
                                        sb.append("Unable to load  text node");
                                        c.b.a.a.a.N((Throwable) obj, sb, "LoaderARContent");
                                        hdVar3.i.removeChild(simpleTransformableNode2);
                                        int i3 = hdVar3.r - 1;
                                        hdVar3.r = i3;
                                        if (i3 == 0) {
                                            hdVar3.k();
                                            return null;
                                        }
                                        return null;
                                    }
                                });
                                Q.setOnAccurateTapListner(new Node.OnAccurateTapListner() { // from class: c.e.b.m5
                                    @Override // com.google.ar.sceneform.Node.OnAccurateTapListner
                                    public final void onAccurateTap(HitTestResult hitTestResult, MotionEvent motionEvent) {
                                        hd hdVar3 = hd.this;
                                        ve veVar3 = veVar2;
                                        MediaPlayer mediaPlayer2 = mediaPlayer;
                                        Objects.requireNonNull(hdVar3);
                                        String str = veVar3.f5355f;
                                        str.hashCode();
                                        char c2 = 65535;
                                        switch (str.hashCode()) {
                                            case -1858525552:
                                                if (str.equals("GOTO_SCENE")) {
                                                    c2 = 0;
                                                    break;
                                                }
                                                break;
                                            case -1235449741:
                                                if (str.equals("GOTO_URL")) {
                                                    c2 = 1;
                                                    break;
                                                }
                                                break;
                                            case -967933916:
                                                if (str.equals("PLAY_SOUND")) {
                                                    c2 = 2;
                                                    break;
                                                }
                                                break;
                                            case -582612232:
                                                if (str.equals("DIAL_NUMBER")) {
                                                    c2 = 3;
                                                    break;
                                                }
                                                break;
                                            case 1125406319:
                                                if (str.equals("COMPOSE_EMAIL")) {
                                                    c2 = 4;
                                                    break;
                                                }
                                                break;
                                        }
                                        switch (c2) {
                                            case 0:
                                                hdVar3.C = veVar3.f5356g;
                                                hdVar3.n();
                                                return;
                                            case 1:
                                                hdVar3.o(veVar3.f5356g);
                                                return;
                                            case 2:
                                                if (mediaPlayer2.isPlaying()) {
                                                    mediaPlayer2.pause();
                                                    return;
                                                } else {
                                                    mediaPlayer2.start();
                                                    return;
                                                }
                                            case 3:
                                                hdVar3.j(veVar3.f5356g);
                                                return;
                                            case 4:
                                                hdVar3.x(veVar3.f5356g);
                                                return;
                                            default:
                                                return;
                                        }
                                    }
                                });
                            }
                        });
                        break;
                    case 11:
                        hdVar.f4816g.runOnUiThread(new Runnable() { // from class: c.e.b.b7
                            @Override // java.lang.Runnable
                            public final void run() {
                                hd hdVar2 = hd.this;
                                ve veVar2 = veVar;
                                Objects.requireNonNull(hdVar2);
                                if (Objects.equals(veVar2.B, "POINT_LIGHT")) {
                                    Light build = Light.builder(Light.Type.POINT).setColor(new com.google.ar.sceneform.rendering.Color(hdVar2.l(veVar2.w))).setShadowCastingEnabled(true).setIntensity(veVar2.F * 300.0f).setFalloffRadius(100.0f).build();
                                    Node node2 = new Node();
                                    float[] fArr = veVar2.l;
                                    node2.setLocalPosition(new Vector3(fArr[0], fArr[1], fArr[2]));
                                    float[] fArr2 = veVar2.q;
                                    node2.setLocalRotation(Quaternion.multiply(Quaternion.eulerAngles(new Vector3(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 180.0f)), new Quaternion(fArr2[0], fArr2[1], fArr2[2], fArr2[3])));
                                    float[] fArr3 = veVar2.n;
                                    c.b.a.a.a.I(new Vector3(fArr3[0], fArr3[1], fArr3[2]), veVar2.p, node2, build);
                                    node2.setParent(hdVar2.i);
                                } else if (Objects.equals(veVar2.B, "SPOT_LIGHT")) {
                                    Light.Builder builder = Light.builder(Light.Type.SPOTLIGHT);
                                    Context context = hdVar2.f4815f;
                                    Object obj = b.j.c.a.f2074a;
                                    Light build2 = builder.setColor(new com.google.ar.sceneform.rendering.Color(context.getColor(R.color.arGalleryAppBarColorDark))).setShadowCastingEnabled(true).setIntensity(veVar2.F * 300.0f).setFalloffRadius(100.0f).build();
                                    Node node3 = new Node();
                                    float[] fArr4 = veVar2.l;
                                    node3.setLocalPosition(new Vector3(fArr4[0], fArr4[1], fArr4[2]));
                                    float[] fArr5 = veVar2.q;
                                    node3.setLocalRotation(Quaternion.multiply(Quaternion.eulerAngles(new Vector3(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 180.0f)), new Quaternion(fArr5[0], fArr5[1], fArr5[2], fArr5[3])));
                                    float[] fArr6 = veVar2.n;
                                    c.b.a.a.a.I(new Vector3(fArr6[0], fArr6[1], fArr6[2]), veVar2.p, node3, build2);
                                    node3.setParent(hdVar2.i);
                                } else if (Objects.equals(veVar2.B, "DIRECTIONAL_LIGHT")) {
                                    Light build3 = Light.builder(Light.Type.DIRECTIONAL).setColor(new com.google.ar.sceneform.rendering.Color(hdVar2.l(veVar2.w))).setShadowCastingEnabled(true).setIntensity(veVar2.F * 300.0f).build();
                                    Node node4 = new Node();
                                    float[] fArr7 = veVar2.l;
                                    node4.setLocalPosition(new Vector3(fArr7[0], fArr7[1], fArr7[2]));
                                    float[] fArr8 = veVar2.q;
                                    node4.setLocalRotation(Quaternion.multiply(Quaternion.eulerAngles(new Vector3(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 180.0f)), new Quaternion(fArr8[0], fArr8[1], fArr8[2], fArr8[3])));
                                    float[] fArr9 = veVar2.n;
                                    c.b.a.a.a.I(new Vector3(fArr9[0], fArr9[1], fArr9[2]), veVar2.p, node4, build3);
                                    node4.setParent(hdVar2.i);
                                }
                                int i3 = hdVar2.r - 1;
                                hdVar2.r = i3;
                                if (i3 == 0) {
                                    hdVar2.k();
                                }
                            }
                        });
                        break;
                    default:
                        Log.d("LoaderARContent", i2 + " is not supported");
                        break;
                }
            } catch (NumberFormatException unused) {
                hdVar.f4816g.runOnUiThread(new ld(hdVar));
            }
        } catch (JSONException e2) {
            e2.printStackTrace();
        }
    }

    public void b(ve veVar, MediaPlayer mediaPlayer, Node node) {
        String str = veVar.f5355f;
        str.hashCode();
        char c2 = 65535;
        switch (str.hashCode()) {
            case -1858525552:
                if (str.equals("GOTO_SCENE")) {
                    c2 = 0;
                    break;
                }
                break;
            case -1856802224:
                if (str.equals("REPLACE_TEXTURE")) {
                    c2 = 1;
                    break;
                }
                break;
            case -1235449741:
                if (str.equals("GOTO_URL")) {
                    c2 = 2;
                    break;
                }
                break;
            case -1141634779:
                if (str.equals("PLAY_PAUSE_CONTENT")) {
                    c2 = 3;
                    break;
                }
                break;
            case -967933916:
                if (str.equals("PLAY_SOUND")) {
                    c2 = 4;
                    break;
                }
                break;
            case -582612232:
                if (str.equals("DIAL_NUMBER")) {
                    c2 = 5;
                    break;
                }
                break;
            case 1125406319:
                if (str.equals("COMPOSE_EMAIL")) {
                    c2 = 6;
                    break;
                }
                break;
        }
        switch (c2) {
            case 0:
                this.C = veVar.f5356g;
                n();
                return;
            case 1:
                NavigableMap navigableMap = this.O.get(Integer.valueOf(veVar.H));
                Iterator it = navigableMap.entrySet().iterator();
                String str2 = "";
                while (true) {
                    if (it.hasNext()) {
                        Map.Entry entry = (Map.Entry) it.next();
                        Map.Entry higherEntry = navigableMap.higherEntry((Integer) entry.getKey());
                        if (Objects.equals(this.N.get(Integer.valueOf(veVar.H)), "default")) {
                            str2 = (String) entry.getValue();
                        } else if (Objects.equals(entry.getValue(), this.N.get(Integer.valueOf(veVar.H)))) {
                            str2 = (String) higherEntry.getValue();
                        }
                    }
                }
                p(this.M.get(Integer.valueOf(veVar.H)), node, str2, veVar.H);
                return;
            case 2:
                o(veVar.f5356g);
                return;
            case 3:
                if (this.H.get(Integer.valueOf(veVar.H)) == null) {
                    return;
                }
                ObjectAnimator objectAnimator = this.E.get(Integer.valueOf(veVar.H));
                if (objectAnimator != null) {
                    if (objectAnimator.getPropertyName().toLowerCase().contains(veVar.z.toLowerCase())) {
                        if (!objectAnimator.isPaused() && objectAnimator.isRunning()) {
                            objectAnimator.pause();
                            return;
                        } else {
                            objectAnimator.resume();
                            return;
                        }
                    } else if (Objects.equals(this.G.get(Integer.valueOf(veVar.H)), "model")) {
                        if (objectAnimator.isPaused()) {
                            objectAnimator.resume();
                            return;
                        } else if (!objectAnimator.isRunning()) {
                            objectAnimator.start();
                            return;
                        } else {
                            objectAnimator.pause();
                            return;
                        }
                    } else {
                        objectAnimator.pause();
                        objectAnimator.removeAllUpdateListeners();
                        objectAnimator.removeAllListeners();
                        objectAnimator.end();
                        objectAnimator.cancel();
                        f(node.getRenderableInstance(), veVar.A, veVar.H, veVar.z, true, "model");
                        return;
                    }
                }
                f(node.getRenderableInstance(), veVar.A, veVar.H, veVar.z, true, "model");
                return;
            case 4:
                if (mediaPlayer.isPlaying()) {
                    mediaPlayer.pause();
                    return;
                } else {
                    mediaPlayer.start();
                    return;
                }
            case 5:
                j(veVar.f5356g);
                return;
            case 6:
                x(veVar.f5356g);
                return;
            default:
                return;
        }
    }

    public final void c(int i) {
        this.o.postDelayed(this.p, i);
        if (this.w) {
            return;
        }
        this.w = true;
        this.l.getArSceneView().getScene().addOnPeekTouchListener(new Scene.OnPeekTouchListener() { // from class: c.e.b.n6
            @Override // com.google.ar.sceneform.Scene.OnPeekTouchListener
            public final void onPeekTouch(HitTestResult hitTestResult, MotionEvent motionEvent) {
                hd hdVar = hd.this;
                if (hdVar.x) {
                    return;
                }
                if (!hdVar.v) {
                    hdVar.z(true);
                } else if (motionEvent.getAction() == 1) {
                    hdVar.o.postDelayed(hdVar.p, 2000L);
                }
            }
        });
    }

    public void d(ve veVar) {
        float f2 = veVar.f5350a;
        int i = veVar.f5351b;
        if (i == 0) {
            Node sunlight = this.l.getArSceneView().getScene().getSunlight();
            Objects.requireNonNull(sunlight);
            Light light = sunlight.getLight();
            Objects.requireNonNull(light);
            light.setIntensity(200.0f * f2);
            StringBuilder sb = new StringBuilder();
            sb.append("Light0 ");
            sb.append(f2);
            sb.append(" ");
            Light light2 = this.l.getArSceneView().getScene().getSunlight().getLight();
            Objects.requireNonNull(light2);
            sb.append(light2.getIntensity());
            Log.d("LoaderARContent", sb.toString());
        } else if (i == 1) {
            Light build = Light.builder(Light.Type.DIRECTIONAL).setColor(new com.google.ar.sceneform.rendering.Color(1.0f, 1.0f, 1.0f)).setShadowCastingEnabled(true).setIntensity(50.0f * f2).build();
            this.l.getArSceneView().getScene().getSunlight().setEnabled(false);
            this.l.getArSceneView().getScene().setLightEstimate(new com.google.ar.sceneform.rendering.Color(1.0f, 1.0f, 1.0f), f2 / 5.0f);
            Node node = new Node();
            node.setWorldPosition(new Vector3(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 10.0f, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD));
            node.setLight(build);
            node.setWorldRotation(Quaternion.eulerAngles(new Vector3(-30.0f, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 30.0f)));
            node.setParent(this.i);
            this.t++;
            StringBuilder x = c.b.a.a.a.x("Light1 ");
            x.append(build.getIntensity());
            x.append(" ");
            x.append(this.l.getArSceneView().getScene().getSunlight().getLight().getIntensity());
            Log.d("LoaderARContent", x.toString());
        } else if (i == 2) {
            Light.Type type = Light.Type.DIRECTIONAL;
            float f3 = 500.0f * f2;
            Light build2 = Light.builder(type).setColor(new com.google.ar.sceneform.rendering.Color(1.0f, 1.0f, 1.0f)).setShadowCastingEnabled(false).setIntensity(f3).build();
            Light build3 = Light.builder(type).setColor(new com.google.ar.sceneform.rendering.Color(1.0f, 1.0f, 1.0f)).setShadowCastingEnabled(false).setIntensity(f3).build();
            this.l.getArSceneView().getScene().getSunlight().setLight(build2);
            this.l.getArSceneView().getScene().setLightEstimate(new com.google.ar.sceneform.rendering.Color(1.0f, 1.0f, 1.0f), f2 / 5.0f);
            Node node2 = new Node();
            node2.setWorldPosition(new Vector3(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 10.0f, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD));
            node2.setWorldRotation(Quaternion.eulerAngles(new Vector3(-30.0f, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 30.0f)));
            node2.setLight(build3);
            node2.setParent(this.i);
            this.t++;
            StringBuilder x2 = c.b.a.a.a.x("Light ");
            x2.append(build2.getIntensity());
            x2.append(" ");
            x2.append(this.l.getArSceneView().getScene().getSunlight().getLight().getIntensity());
            Log.d("LoaderARContent", x2.toString());
        }
    }

    public void e(final Node node, ve veVar, int i) {
        String str;
        Vector3 vector3;
        Vector3 vector32;
        Vector3 vector33;
        Vector3 vector34;
        Vector3 vector35;
        Vector3 localPosition;
        Vector3 vector36;
        Vector3 localScale;
        Vector3 vector37;
        Vector3 localPosition2;
        Vector3 vector38;
        Vector3 localPosition3;
        if (i == 0) {
            str = veVar.E;
        } else {
            str = i == 1 ? veVar.D : "NO_TRANSITION";
        }
        if ((Objects.equals(str, "NO_TRANSITION") || Objects.equals(str, "")) && i == 1) {
            s(node);
        } else if (Objects.equals(str, "NO_TRANSITION")) {
        } else {
            ObjectAnimator objectAnimator = new ObjectAnimator();
            objectAnimator.setInterpolator(new LinearInterpolator());
            objectAnimator.addListener(new b(i, node));
            str.hashCode();
            char c2 = 65535;
            switch (str.hashCode()) {
                case -1856508265:
                    if (str.equals("SCALE_DOWN")) {
                        c2 = 0;
                        break;
                    }
                    break;
                case -1810415154:
                    if (str.equals("SLIDE_RIGHT")) {
                        c2 = 1;
                        break;
                    }
                    break;
                case -489950199:
                    if (str.equals("SLIDE_UP")) {
                        c2 = 2;
                        break;
                    }
                    break;
                case -373408312:
                    if (str.equals("FADE_IN")) {
                        c2 = 3;
                        break;
                    }
                    break;
                case -109193776:
                    if (str.equals("SCALE_UP")) {
                        c2 = 4;
                        break;
                    }
                    break;
                case 1309250283:
                    if (str.equals("FADE_OUT")) {
                        c2 = 5;
                        break;
                    }
                    break;
                case 1603756688:
                    if (str.equals("SLIDE_DOWN")) {
                        c2 = 6;
                        break;
                    }
                    break;
                case 1603984885:
                    if (str.equals("SLIDE_LEFT")) {
                        c2 = 7;
                        break;
                    }
                    break;
                case 1965091464:
                    if (str.equals("BOUNCE")) {
                        c2 = '\b';
                        break;
                    }
                    break;
            }
            switch (c2) {
                case 0:
                    if (i == 1) {
                        vector3 = node.getLocalScale();
                        vector32 = new Vector3(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
                    } else {
                        vector3 = new Vector3(2.0f, 2.0f, 2.0f);
                        if (!Objects.equals(veVar.I, "AR Button") && !Objects.equals(veVar.I, "ARText")) {
                            vector32 = node.getLocalScale();
                        } else {
                            vector32 = new Vector3(0.64f, 0.64f, 0.64f);
                        }
                    }
                    objectAnimator.setObjectValues(vector3, vector32);
                    objectAnimator.setPropertyName("localScale");
                    objectAnimator.setEvaluator(new Vector3Evaluator());
                    break;
                case 1:
                    if (i == 1) {
                        vector33 = node.getLocalPosition();
                        vector34 = new Vector3(1.0f, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
                    } else {
                        vector33 = new Vector3(-1.0f, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
                        vector34 = new Vector3(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
                    }
                    objectAnimator.setObjectValues(vector33, vector34);
                    objectAnimator.setPropertyName("localPosition");
                    objectAnimator.setEvaluator(new Vector3Evaluator());
                    break;
                case 2:
                    if (i == 1) {
                        vector35 = node.getLocalPosition();
                        localPosition = new Vector3(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 1.0f, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
                    } else {
                        vector35 = new Vector3(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, -1.0f, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
                        localPosition = node.getLocalPosition();
                    }
                    objectAnimator.setObjectValues(vector35, localPosition);
                    objectAnimator.setPropertyName("localPosition");
                    objectAnimator.setEvaluator(new Vector3Evaluator());
                    break;
                case 3:
                    objectAnimator.setFloatValues(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 1.0f);
                    objectAnimator.setPropertyName("alpha");
                    break;
                case 4:
                    if (i == 1) {
                        vector36 = new Vector3(1.0f, 1.0f, 1.0f);
                        localScale = new Vector3(2.0f, 2.0f, 2.0f);
                    } else {
                        vector36 = new Vector3(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
                        localScale = node.getLocalScale();
                    }
                    objectAnimator.setObjectValues(vector36, localScale);
                    objectAnimator.setPropertyName("localScale");
                    objectAnimator.setEvaluator(new Vector3Evaluator());
                    break;
                case 5:
                    objectAnimator.setFloatValues(1.0f, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
                    objectAnimator.setPropertyName("alpha");
                    break;
                case 6:
                    if (i == 1) {
                        vector37 = node.getLocalPosition();
                        localPosition2 = new Vector3(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, -1.0f, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
                    } else {
                        vector37 = new Vector3(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 1.0f, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
                        localPosition2 = node.getLocalPosition();
                    }
                    objectAnimator.setObjectValues(vector37, localPosition2);
                    objectAnimator.setPropertyName("localPosition");
                    objectAnimator.setEvaluator(new Vector3Evaluator());
                    break;
                case 7:
                    if (i == 1) {
                        vector38 = node.getLocalPosition();
                        localPosition3 = new Vector3(-1.0f, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
                    } else {
                        vector38 = new Vector3(1.0f, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
                        localPosition3 = node.getLocalPosition();
                    }
                    objectAnimator.setObjectValues(vector38, localPosition3);
                    objectAnimator.setPropertyName("localPosition");
                    objectAnimator.setEvaluator(new Vector3Evaluator());
                    break;
                case '\b':
                    objectAnimator.setFloatValues(2.0f, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
                    objectAnimator.setInterpolator(new BounceInterpolator());
                    objectAnimator.addUpdateListener(new ValueAnimator.AnimatorUpdateListener() { // from class: c.e.b.t4
                        @Override // android.animation.ValueAnimator.AnimatorUpdateListener
                        public final void onAnimationUpdate(ValueAnimator valueAnimator) {
                            Node.this.setLocalPosition(new Vector3(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, ((Float) valueAnimator.getAnimatedValue()).floatValue()));
                        }
                    });
                    break;
                default:
                    return;
            }
            if (!str.equals("FADE_IN") && !str.equals("FADE_OUT")) {
                objectAnimator.setTarget(node);
            } else {
                Renderable renderable = node.getRenderable();
                Objects.requireNonNull(renderable);
                objectAnimator.setTarget(((ViewRenderable) renderable).getView());
            }
            objectAnimator.setDuration((long) (veVar.C * 1000.0d));
            objectAnimator.start();
        }
    }

    public final void f(RenderableInstance renderableInstance, String str, int i, String str2, boolean z, String str3) {
        this.G.put(Integer.valueOf(i), str3);
        this.F.put(Integer.valueOf(i), new String[]{str2, str});
        if (str2.equals("IDLE") || str2.length() == 0 || renderableInstance.getFilamentAsset().getAnimator().getAnimationCount() == 0) {
            return;
        }
        int animationCount = renderableInstance.getAnimationCount();
        this.T = renderableInstance.animate(0);
        if (str.equals("REPEAT") && animationCount == 0) {
            this.T.setRepeatCount(-1);
        } else {
            this.T.setRepeatCount(0);
        }
        if (z && str2.equals("ALL")) {
            this.T.start();
        } else {
            this.T.pause();
        }
        if (str2.equals("ALL")) {
            this.T.addListener(new e(i, new int[]{0}, animationCount, str, renderableInstance, str3));
        } else {
            this.T = renderableInstance.animate(str2);
            if (str.equals("REPEAT") && animationCount > 0) {
                this.T.setRepeatCount(-1);
            } else {
                this.T.setRepeatCount(0);
            }
            if (z) {
                this.T.start();
            } else {
                this.T.pause();
            }
            if (Objects.equals(str3, "trigger")) {
                this.T.addListener(new a(i));
            }
        }
        this.E.put(Integer.valueOf(i), this.T);
    }

    public void g(ve veVar, MediaPlayer mediaPlayer) {
        try {
            mediaPlayer.setDataSource(veVar.f5356g);
            mediaPlayer.prepare();
        } catch (Exception e2) {
            e2.printStackTrace();
        }
    }

    public void h() {
        this.x = true;
        if (this.y.length() == 0 || !this.u) {
            return;
        }
        long currentTimeMillis = (System.currentTimeMillis() - this.q) / 1000;
        ec ecVar = new ec();
        String v = c.b.a.a.a.v(new StringBuilder(), this.z, "unitear_app/save_scan_spend_time");
        String string = Settings.Secure.getString(this.f4815f.getContentResolver(), "android_id");
        u.a aVar = new u.a();
        aVar.c(f.u.f6108b);
        aVar.a("campaign_category_id", "1");
        aVar.a("unique_id", this.y);
        aVar.a("time_spend", "" + currentTimeMillis);
        aVar.a("scan_mode", "1");
        aVar.a("device_id", string);
        f.u b2 = aVar.b();
        StringBuilder A = c.b.a.a.a.A(v, "  ");
        A.append(this.y);
        A.append(" time = ");
        A.append(currentTimeMillis);
        Log.d("LoaderARContent", A.toString());
        ecVar.a(v, b2, new id(this));
    }

    public void i() {
        final ObjectAnimator objectAnimator = new ObjectAnimator();
        objectAnimator.setInterpolator(new LinearInterpolator());
        Node node = new Node();
        this.j = node;
        node.setParent(this.i);
        this.j.setLocalPosition(new Vector3(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 0.02f));
        this.j.setName("loading_node");
        ViewRenderable.builder().setView(this.f4816g, R.layout.image_target_loader).build().thenAccept(new Consumer() { // from class: c.e.b.l5
            @Override // java.util.function.Consumer
            public final void accept(Object obj) {
                hd hdVar = hd.this;
                ObjectAnimator objectAnimator2 = objectAnimator;
                ViewRenderable viewRenderable = (ViewRenderable) obj;
                Objects.requireNonNull(hdVar);
                ((ProgressBar) viewRenderable.getView().findViewById(R.id.imageTargetLoaderProgressbar)).setIndeterminate(true);
                ((TextView) viewRenderable.getView().findViewById(R.id.imageTargetLoaderProgressbarText)).setText("");
                viewRenderable.setRenderPriority(7);
                viewRenderable.setHorizontalAlignment(ViewRenderable.HorizontalAlignment.CENTER);
                viewRenderable.setVerticalAlignment(ViewRenderable.VerticalAlignment.CENTER);
                c.b.a.a.a.C(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, hdVar.j);
                objectAnimator2.setObjectValues(hdVar.j.getLocalScale(), new Vector3(0.12f, 0.12f, 0.12f));
                objectAnimator2.setPropertyName("localScale");
                objectAnimator2.setEvaluator(new Vector3Evaluator());
                objectAnimator2.setTarget(hdVar.j);
                objectAnimator2.setDuration(1000L);
                objectAnimator2.start();
                hdVar.j.setRenderable(viewRenderable);
            }
        }).exceptionally(new Function() { // from class: c.e.b.k7
            @Override // java.util.function.Function
            public final Object apply(Object obj) {
                Throwable th = (Throwable) obj;
                Objects.requireNonNull(hd.this);
                Log.e("LoaderARContent", "Unable to load  createImageTargetLoader");
                return null;
            }
        });
    }

    public final void j(String str) {
        this.f4815f.startActivity(new Intent("android.intent.action.DIAL", Uri.parse("tel:" + str)));
    }

    public void k() {
        if (this.j.isEnabled() && this.r == 0 && this.D) {
            ObjectAnimator objectAnimator = new ObjectAnimator();
            objectAnimator.setInterpolator(new LinearInterpolator());
            objectAnimator.setObjectValues(this.j.getLocalScale(), new Vector3(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD));
            objectAnimator.setPropertyName("localScale");
            objectAnimator.setEvaluator(new Vector3Evaluator());
            objectAnimator.setTarget(this.j);
            objectAnimator.setDuration(500L);
            objectAnimator.start();
            objectAnimator.addListener(new c());
        }
    }

    public final int l(String str) {
        if (str.length() > 8) {
            str = c.b.a.a.a.r("#", str.substring(str.length() - 2), str.substring(1, str.length() - 2));
        }
        return Color.parseColor(str);
    }

    public final MediaPlayer m() {
        if (f4810a == null) {
            MediaPlayer mediaPlayer = new MediaPlayer();
            f4810a = mediaPlayer;
            return mediaPlayer;
        } else if (f4811b == null) {
            MediaPlayer mediaPlayer2 = new MediaPlayer();
            f4811b = mediaPlayer2;
            return mediaPlayer2;
        } else if (f4812c == null) {
            MediaPlayer mediaPlayer3 = new MediaPlayer();
            f4812c = mediaPlayer3;
            return mediaPlayer3;
        } else if (f4813d == null) {
            MediaPlayer mediaPlayer4 = new MediaPlayer();
            f4813d = mediaPlayer4;
            return mediaPlayer4;
        } else {
            MediaPlayer mediaPlayer5 = new MediaPlayer();
            f4814e = mediaPlayer5;
            return mediaPlayer5;
        }
    }

    public final void n() {
        y();
        this.L.forEach(new BiConsumer() { // from class: c.e.b.i5
            @Override // java.util.function.BiConsumer
            public final void accept(Object obj, Object obj2) {
                hd.this.e((Node) obj, (ve) obj2, 1);
            }
        });
    }

    public final void o(String str) {
        if (!str.startsWith("http://") && !str.startsWith("https://")) {
            str = c.b.a.a.a.q("http://", str);
        }
        this.f4815f.startActivity(new Intent("android.intent.action.VIEW", Uri.parse(str)));
    }

    public void p(ModelRenderable modelRenderable, final Node node, final String str, final int i) {
        if (Objects.equals(this.N.get(Integer.valueOf(i)), str)) {
            return;
        }
        if (!this.M.containsKey(Integer.valueOf(i))) {
            this.M.put(Integer.valueOf(i), modelRenderable);
        }
        if (!Objects.equals(str, "default") && str != null) {
            Texture.builder().setSampler(Texture.Sampler.builder().setMinFilter(Texture.Sampler.MinFilter.LINEAR_MIPMAP_LINEAR).setMagFilter(Texture.Sampler.MagFilter.LINEAR).setWrapMode(Texture.Sampler.WrapMode.REPEAT).build()).setSource(this.f4815f, Uri.parse(this.m.a(str.substring(str.lastIndexOf(47) + 1)))).setUsage(Texture.Usage.DATA).build().thenAccept(new Consumer() { // from class: c.e.b.d6
                @Override // java.util.function.Consumer
                public final void accept(Object obj) {
                    hd hdVar = hd.this;
                    Node node2 = node;
                    int i2 = i;
                    String str2 = str;
                    Texture texture = (Texture) obj;
                    Objects.requireNonNull(hdVar);
                    RenderableInstance renderableInstance = node2.getRenderableInstance();
                    int materialsCount = renderableInstance.getMaterialsCount();
                    for (int i3 = 0; i3 < materialsCount; i3++) {
                        renderableInstance.getMaterial(i3).setTexture("baseColorMap", texture);
                    }
                    hdVar.N.put(Integer.valueOf(i2), str2);
                    if (hdVar.E.get(Integer.valueOf(i2)) != null && !hdVar.F.get(Integer.valueOf(i2))[0].toLowerCase().equals("all")) {
                        node2.getRenderableInstance().animate(hdVar.F.get(Integer.valueOf(i2))[0]).start();
                        return;
                    }
                    Node node3 = hdVar.H.get(Integer.valueOf(i2));
                    Objects.requireNonNull(node3);
                    RenderableInstance renderableInstance2 = node3.getRenderableInstance();
                    ve veVar = hdVar.I.get(Integer.valueOf(i2));
                    Objects.requireNonNull(veVar);
                    int i4 = veVar.H;
                    ve veVar2 = hdVar.I.get(Integer.valueOf(i2));
                    Objects.requireNonNull(veVar2);
                    String str3 = veVar2.z;
                    ve veVar3 = hdVar.I.get(Integer.valueOf(i2));
                    Objects.requireNonNull(veVar3);
                    hdVar.f(renderableInstance2, "REPEAT", i4, str3, veVar3.j, "model");
                }
            }).exceptionally(new Function() { // from class: c.e.b.v6
                @Override // java.util.function.Function
                public final Object apply(Object obj) {
                    Throwable th = (Throwable) obj;
                    Objects.requireNonNull(hd.this);
                    Log.e("LoaderARContent", "Unable to load texture");
                    return null;
                }
            });
            return;
        }
        ModelRenderable modelRenderable2 = this.M.get(Integer.valueOf(i));
        Objects.requireNonNull(modelRenderable2);
        node.setRenderable(modelRenderable2.makeCopy());
        this.N.put(Integer.valueOf(i), "default");
        if (node.getRenderableInstance() != null && this.E.get(Integer.valueOf(i)) != null) {
            String[] strArr = this.F.get(Integer.valueOf(i));
            Objects.requireNonNull(strArr);
            if (!strArr[0].toLowerCase().equals("all")) {
                RenderableInstance renderableInstance = node.getRenderableInstance();
                String[] strArr2 = this.F.get(Integer.valueOf(i));
                Objects.requireNonNull(strArr2);
                renderableInstance.animate(strArr2[0]).start();
                return;
            }
        }
        Node node2 = this.H.get(Integer.valueOf(i));
        Objects.requireNonNull(node2);
        RenderableInstance renderableInstance2 = node2.getRenderableInstance();
        ve veVar = this.I.get(Integer.valueOf(i));
        Objects.requireNonNull(veVar);
        int i2 = veVar.H;
        ve veVar2 = this.I.get(Integer.valueOf(i));
        Objects.requireNonNull(veVar2);
        String str2 = veVar2.z;
        ve veVar3 = this.I.get(Integer.valueOf(i));
        Objects.requireNonNull(veVar3);
        f(renderableInstance2, "REPEAT", i2, str2, veVar3.j, "model");
    }

    public void q(Node node, final int i, final int[] iArr, final String[] strArr, final ImageView imageView, final Node node2, final Node node3) {
        float f2 = i * 0.35f;
        Node node4 = i > 0 ? node3 : node2;
        this.n.add(node4);
        node4.setParent(node);
        node4.setLocalPosition(new Vector3(f2, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 0.01f));
        node4.setOnAccurateTapListner(new Node.OnAccurateTapListner() { // from class: c.e.b.e4
            @Override // com.google.ar.sceneform.Node.OnAccurateTapListner
            public final void onAccurateTap(HitTestResult hitTestResult, MotionEvent motionEvent) {
                hd hdVar = hd.this;
                int[] iArr2 = iArr;
                String[] strArr2 = strArr;
                Node node5 = node2;
                Node node6 = node3;
                ImageView imageView2 = imageView;
                Objects.requireNonNull(hdVar);
                iArr2[0] = iArr2[0] + 1;
                if (iArr2[0] < 0) {
                    iArr2[0] = strArr2.length - 1;
                }
                if (iArr2[0] > strArr2.length - 1) {
                    iArr2[0] = 0;
                }
                c.c.a.b.d(hdVar.f4816g).k(strArr2[iArr2[0]]).C(new pd(hdVar, node5, node6)).B(imageView2);
            }
        });
        final Node node5 = node4;
        ViewRenderable.builder().setView(this.f4816g, R.layout.img_view).build().thenAccept(new Consumer() { // from class: c.e.b.n5
            @Override // java.util.function.Consumer
            public final void accept(Object obj) {
                hd hdVar = hd.this;
                Node node6 = node5;
                int i2 = i;
                Node node7 = node2;
                Node node8 = node3;
                ViewRenderable viewRenderable = (ViewRenderable) obj;
                Objects.requireNonNull(hdVar);
                ((ImageView) viewRenderable.getView().findViewById(R.id.img_loader_view)).setImageResource(R.drawable.slide_button);
                viewRenderable.setRenderPriority(5);
                viewRenderable.setHorizontalAlignment(ViewRenderable.HorizontalAlignment.CENTER);
                viewRenderable.setVerticalAlignment(ViewRenderable.VerticalAlignment.CENTER);
                node6.setRenderable(viewRenderable);
                node6.setLocalScale(new Vector3(i2 * 0.1f, 0.1f, 0.1f));
                node7.setLocalPosition(new Vector3(-hdVar.U, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 0.01f));
                node8.setLocalPosition(new Vector3(hdVar.U, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 0.01f));
            }
        }).exceptionally(new Function() { // from class: c.e.b.g6
            @Override // java.util.function.Function
            public final Object apply(Object obj) {
                hd hdVar = hd.this;
                Throwable th = (Throwable) obj;
                Objects.requireNonNull(hdVar);
                Log.i("kkkkkkk", "next entered failed");
                Log.e("LoaderARContent", "Unable to load  nextPrevButton");
                int i2 = hdVar.r - 1;
                hdVar.r = i2;
                if (i2 == 0) {
                    hdVar.k();
                    return null;
                }
                return null;
            }
        });
    }

    public void r() {
        MediaPlayer mediaPlayer = f4810a;
        if (mediaPlayer != null && mediaPlayer.isPlaying()) {
            f4810a.pause();
            this.K.put(f4810a, Boolean.TRUE);
        }
        MediaPlayer mediaPlayer2 = f4811b;
        if (mediaPlayer2 != null && mediaPlayer2.isPlaying()) {
            f4811b.pause();
            this.K.put(f4811b, Boolean.TRUE);
        }
        MediaPlayer mediaPlayer3 = f4812c;
        if (mediaPlayer3 != null && mediaPlayer3.isPlaying()) {
            f4812c.pause();
            this.K.put(f4812c, Boolean.TRUE);
        }
        MediaPlayer mediaPlayer4 = f4813d;
        if (mediaPlayer4 != null && mediaPlayer4.isPlaying()) {
            f4813d.pause();
            this.K.put(f4813d, Boolean.TRUE);
        }
        MediaPlayer mediaPlayer5 = f4814e;
        if (mediaPlayer5 == null || !mediaPlayer5.isPlaying()) {
            return;
        }
        f4814e.pause();
        this.K.put(f4814e, Boolean.TRUE);
    }

    public final void s(Node node) {
        this.i.removeChild(node.getParent());
        if (this.i.getChildren().size() == this.t) {
            if (this.C.length() == 0) {
                this.k.a("Scene not found");
                return;
            }
            this.J.clear();
            u(this.k);
        }
    }

    public final void t(String str, Node node, ve veVar, SimpleTransformableNode simpleTransformableNode, Node node2) {
        Log.d("LoaderARContent", "progress complete " + str);
        new c.e.b.p000if.m(this.f4815f, new d(veVar, str, node, node2, simpleTransformableNode)).execute(veVar);
    }

    public void u(g gVar) {
        this.k = gVar;
        String str = !this.u ? "1" : CrashlyticsReportDataCapture.SIGNAL_DEFAULT;
        i();
        this.f4817h = new cc();
        String str2 = this.z + "app/get-ar-new-content/" + this.y + "/" + this.A + "/1/" + str + "/1/" + this.C;
        Log.d("LoaderARContent", str2);
        this.f4817h.a(str2, new kd(this, gVar));
        this.q = System.currentTimeMillis();
    }

    public void v() {
        this.K.forEach(s5.f5213a);
        this.K.clear();
    }

    public void w(Node node, ve veVar) {
        FilamentAsset filamentAsset = node.getRenderableInstance().getFilamentAsset();
        if (filamentAsset != null) {
            Box boundingBox = filamentAsset.getBoundingBox();
            float[] halfExtent = boundingBox.getHalfExtent();
            float[] center = boundingBox.getCenter();
            StringBuilder x = c.b.a.a.a.x("load3Dmodel center ");
            x.append(center[0]);
            x.append(", ");
            x.append(center[1]);
            x.append(", ");
            x.append(center[2]);
            Log.d("LoaderARContent", x.toString());
            float max = Math.max(Math.max(halfExtent[0], halfExtent[1]), halfExtent[2]);
            float f2 = 1.0f / max;
            float f3 = -f2;
            node.setLocalScale(new Vector3(f3, f3, f2));
            Log.d("LoaderARContent", "load3Dmodell bounds " + halfExtent[0] + ", " + halfExtent[1] + ", " + halfExtent[2] + "  Scale = " + f2 + "  MaxBounds = " + max);
            if (veVar == null) {
                return;
            }
            float f4 = center[0] * f2;
            float f5 = center[1] * f2;
            float f6 = center[2] * f2;
            float[] fArr = veVar.l;
            if (fArr[2] == 0.008f || fArr[2] == StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                f5 = (center[1] * f2) - (halfExtent[1] * f2);
            }
            StringBuilder x2 = c.b.a.a.a.x("load3Dmodel yCorrection ");
            x2.append(node.getLocalPosition());
            x2.append(" correction = ");
            x2.append(f4);
            x2.append(", ");
            x2.append(f5);
            x2.append(", ");
            x2.append(f6);
            Log.d("LoaderARContent", x2.toString());
            node.setLocalPosition(new Vector3(node.getLocalPosition().x - f4, node.getLocalPosition().y + f5, node.getLocalPosition().z + f6));
        }
    }

    public final void x(String str) {
        Intent intent = new Intent("android.intent.action.SEND");
        intent.setType("text/plain");
        intent.putExtra("android.intent.extra.EMAIL", new String[]{str});
        try {
            this.f4815f.startActivity(intent);
        } catch (Exception e2) {
            Log.e("LoaderARContent", e2.toString());
            this.f4815f.startActivity(Intent.createChooser(intent, "Send Email"));
        }
    }

    public void y() {
        MediaPlayer mediaPlayer = f4810a;
        if (mediaPlayer != null) {
            mediaPlayer.stop();
            f4810a.release();
            f4810a = null;
        }
        MediaPlayer mediaPlayer2 = f4811b;
        if (mediaPlayer2 != null) {
            mediaPlayer2.stop();
            f4811b.release();
            f4811b = null;
        }
        MediaPlayer mediaPlayer3 = f4812c;
        if (mediaPlayer3 != null) {
            mediaPlayer3.stop();
            f4812c.release();
            f4812c = null;
        }
        MediaPlayer mediaPlayer4 = f4813d;
        if (mediaPlayer4 != null) {
            mediaPlayer4.stop();
            f4813d.release();
            f4813d = null;
        }
        MediaPlayer mediaPlayer5 = f4814e;
        if (mediaPlayer5 != null) {
            mediaPlayer5.stop();
            f4814e.release();
            f4814e = null;
        }
    }

    public final void z(boolean z) {
        Iterator<Node> it = this.n.iterator();
        while (it.hasNext()) {
            it.next().setEnabled(z);
        }
        this.v = z;
    }
}