package c.e.b;

import android.animation.Animator;
import android.animation.AnimatorListenerAdapter;
import android.animation.ObjectAnimator;
import android.animation.ValueAnimator;
import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Context;
import android.content.Intent;
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
import c.e.b.yd;
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
import com.google.ar.sceneform.rendering.Color;
import com.google.ar.sceneform.rendering.ExternalTexture;
import com.google.ar.sceneform.rendering.Light;
import com.google.ar.sceneform.rendering.Material;
import com.google.ar.sceneform.rendering.ModelRenderable;
import com.google.ar.sceneform.rendering.Renderable;
import com.google.ar.sceneform.rendering.RenderableInstance;
import com.google.ar.sceneform.rendering.ShapeFactory;
import com.google.ar.sceneform.rendering.Texture;
import com.google.ar.sceneform.rendering.ViewRenderable;
import com.google.ar.sceneform.ux.ArFragment;
import com.google.ar.sceneform.ux.SimpleTransformableNode;
import com.google.firebase.crashlytics.internal.common.CrashlyticsReportDataCapture;
import com.ibosoninnov.unitear.Player360Activity;
import com.ibosoninnov.unitear.R;
import com.ibosoninnov.unitear.VideoActivity;
import com.ibosoninnov.unitear.YoutubeView;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.NavigableMap;
import java.util.Objects;
import java.util.Timer;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionStage;
import java.util.function.BiConsumer;
import java.util.function.Consumer;
import java.util.function.Function;
import org.json.JSONException;
import org.json.JSONObject;

/* compiled from: LoaderARContentSceneformARCore.java */
/* loaded from: classes2.dex */
public class yd {
    public final String A;
    public String B;
    public ObjectAnimator D;
    public ImageView P;

    /* renamed from: a  reason: collision with root package name */
    public final Context f5450a;

    /* renamed from: b  reason: collision with root package name */
    public final Activity f5451b;

    /* renamed from: c  reason: collision with root package name */
    public cc f5452c;

    /* renamed from: d  reason: collision with root package name */
    public final Node f5453d;

    /* renamed from: e  reason: collision with root package name */
    public Node f5454e;

    /* renamed from: f  reason: collision with root package name */
    public e f5455f;

    /* renamed from: g  reason: collision with root package name */
    public final ArFragment f5456g;

    /* renamed from: h  reason: collision with root package name */
    public c.e.b.p000if.e f5457h;
    public float i;
    public MediaPlayer j;
    public MediaPlayer k;
    public MediaPlayer l;
    public MediaPlayer m;
    public MediaPlayer n;
    public long r;
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
    public final Map<MediaPlayer, Boolean> I = new HashMap();
    public final Map<MediaPlayer, Boolean> J = new HashMap();
    public final Map<Node, ve> K = new HashMap();
    public final Map<Integer, ModelRenderable> L = new HashMap();
    public final Map<Integer, String> M = new HashMap();
    public final Map<Integer, ve> N = new HashMap();
    public Map<Integer, NavigableMap> O = new HashMap();
    public final ArrayList<Node> o = new ArrayList<>();
    public final Handler p = new Handler();
    public final Runnable q = new Runnable() { // from class: c.e.b.z8
        @Override // java.lang.Runnable
        public final void run() {
            yd.this.x(false);
        }
    };

    /* compiled from: LoaderARContentSceneformARCore.java */
    /* loaded from: classes2.dex */
    public class a implements c.e.b.gf.c {

        /* renamed from: a  reason: collision with root package name */
        public final /* synthetic */ ve f5458a;

        /* renamed from: b  reason: collision with root package name */
        public final /* synthetic */ String f5459b;

        /* renamed from: c  reason: collision with root package name */
        public final /* synthetic */ Node f5460c;

        /* renamed from: d  reason: collision with root package name */
        public final /* synthetic */ Node f5461d;

        public a(ve veVar, String str, Node node, Node node2) {
            this.f5458a = veVar;
            this.f5459b = str;
            this.f5460c = node;
            this.f5461d = node2;
        }

        @Override // c.e.b.gf.c
        public void a(String str, int i, String str2) {
        }

        @Override // c.e.b.gf.c
        public void b(String str, String str2) {
            ve veVar = this.f5458a;
            int i = veVar.f5352c;
            if (i != 0) {
                String str3 = (String) veVar.T.get(Integer.valueOf(i));
                Texture.Builder sampler = Texture.builder().setSampler(Texture.Sampler.builder().setMinFilter(Texture.Sampler.MinFilter.LINEAR_MIPMAP_LINEAR).setMagFilter(Texture.Sampler.MagFilter.LINEAR).setWrapMode(Texture.Sampler.WrapMode.REPEAT).build());
                yd ydVar = yd.this;
                CompletableFuture<Texture> build = sampler.setSource(ydVar.f5450a, Uri.parse(ydVar.f5457h.a(str3.substring(str3.lastIndexOf(47) + 1)))).setUsage(Texture.Usage.DATA).build();
                CompletableFuture<ModelRenderable> build2 = ModelRenderable.builder().setSource(yd.this.f5451b, Uri.parse(this.f5459b)).setIsFilamentGltf(true).build();
                final ve veVar2 = this.f5458a;
                final Node node = this.f5460c;
                final Node node2 = this.f5461d;
                build2.thenAcceptBoth((CompletionStage) build, new BiConsumer() { // from class: c.e.b.i8
                    @Override // java.util.function.BiConsumer
                    public final void accept(Object obj, Object obj2) {
                        yd.a aVar = yd.a.this;
                        ve veVar3 = veVar2;
                        Node node3 = node;
                        Node node4 = node2;
                        ModelRenderable modelRenderable = (ModelRenderable) obj;
                        Texture texture = (Texture) obj2;
                        Objects.requireNonNull(aVar);
                        Log.d("LoaderARContentSceneformARCore", "load3Dmodel model loaded");
                        yd.this.L.put(Integer.valueOf(veVar3.H), modelRenderable.makeCopy());
                        RenderableInstance renderable = node3.setRenderable(modelRenderable);
                        int materialsCount = renderable.getMaterialsCount();
                        for (int i2 = 0; i2 < materialsCount; i2++) {
                            renderable.getMaterial(i2).setTexture("baseColorMap", texture);
                        }
                        yd.this.M.put(Integer.valueOf(veVar3.H), "default");
                        node3.setRenderable(modelRenderable);
                        yd.this.e(node3.getRenderableInstance(), veVar3.A, veVar3.H, veVar3.z, veVar3.j, "model");
                        yd.this.u(node3, veVar3);
                        yd.this.c(veVar3);
                        yd.this.d(node3, veVar3, 0);
                        yd.this.K.put(node4, veVar3);
                        yd.this.H.put(Integer.valueOf(veVar3.H), node3);
                        yd ydVar2 = yd.this;
                        int i3 = ydVar2.s - 1;
                        ydVar2.s = i3;
                        if (i3 == 0) {
                            ydVar2.j();
                        }
                    }
                }).exceptionally(new Function() { // from class: c.e.b.f8
                    @Override // java.util.function.Function
                    public final Object apply(Object obj) {
                        Objects.requireNonNull(yd.a.this);
                        StringBuilder sb = new StringBuilder();
                        sb.append("load3Dmodel ++");
                        c.b.a.a.a.N((Throwable) obj, sb, "LoaderARContentSceneformARCore");
                        return null;
                    }
                });
                return;
            }
            CompletableFuture<ModelRenderable> build3 = ModelRenderable.builder().setSource(yd.this.f5451b, Uri.parse(this.f5459b)).setIsFilamentGltf(true).build();
            final Node node3 = this.f5460c;
            final ve veVar3 = this.f5458a;
            final Node node4 = this.f5461d;
            build3.thenAccept(new Consumer() { // from class: c.e.b.h8
                @Override // java.util.function.Consumer
                public final void accept(Object obj) {
                    yd.a aVar = yd.a.this;
                    Node node5 = node3;
                    ve veVar4 = veVar3;
                    Node node6 = node4;
                    Objects.requireNonNull(aVar);
                    Log.d("LoaderARContentSceneformARCore", "load3Dmodel model loaded");
                    node5.setRenderable((ModelRenderable) obj);
                    yd.this.e(node5.getRenderableInstance(), veVar4.A, veVar4.H, veVar4.z, veVar4.j, "model");
                    yd.this.u(node5, veVar4);
                    yd.this.c(veVar4);
                    yd.this.d(node5, veVar4, 0);
                    yd.this.K.put(node6, veVar4);
                    yd.this.H.put(Integer.valueOf(veVar4.H), node5);
                    yd ydVar2 = yd.this;
                    int i2 = ydVar2.s - 1;
                    ydVar2.s = i2;
                    if (i2 == 0) {
                        ydVar2.j();
                    }
                }
            }).exceptionally(new Function() { // from class: c.e.b.g8
                @Override // java.util.function.Function
                public final Object apply(Object obj) {
                    Objects.requireNonNull(yd.a.this);
                    StringBuilder sb = new StringBuilder();
                    sb.append("load3Dmodel --");
                    c.b.a.a.a.N((Throwable) obj, sb, "LoaderARContentSceneformARCore");
                    return null;
                }
            });
        }
    }

    /* compiled from: LoaderARContentSceneformARCore.java */
    /* loaded from: classes2.dex */
    public class b extends AnimatorListenerAdapter {

        /* renamed from: a  reason: collision with root package name */
        public final /* synthetic */ int f5463a;

        /* renamed from: b  reason: collision with root package name */
        public final /* synthetic */ int[] f5464b;

        /* renamed from: c  reason: collision with root package name */
        public final /* synthetic */ int f5465c;

        /* renamed from: d  reason: collision with root package name */
        public final /* synthetic */ String f5466d;

        /* renamed from: e  reason: collision with root package name */
        public final /* synthetic */ RenderableInstance f5467e;

        /* renamed from: f  reason: collision with root package name */
        public final /* synthetic */ String f5468f;

        public b(int i, int[] iArr, int i2, String str, RenderableInstance renderableInstance, String str2) {
            this.f5463a = i;
            this.f5464b = iArr;
            this.f5465c = i2;
            this.f5466d = str;
            this.f5467e = renderableInstance;
            this.f5468f = str2;
        }

        @Override // android.animation.AnimatorListenerAdapter, android.animation.Animator.AnimatorListener
        public void onAnimationEnd(Animator animator) {
            super.onAnimationEnd(animator);
            yd ydVar = yd.this;
            ydVar.D = ydVar.E.get(Integer.valueOf(this.f5463a));
            int[] iArr = this.f5464b;
            iArr[0] = iArr[0] + 1;
            if (iArr[0] == this.f5465c) {
                if (this.f5466d.equals("REPEAT")) {
                    int[] iArr2 = this.f5464b;
                    iArr2[0] = 0;
                    yd.this.D = this.f5467e.animate(iArr2[0]);
                    yd.this.D.setRepeatCount(0);
                    yd.this.D.addListener(this);
                    yd.this.D.start();
                    yd.this.E.put(Integer.valueOf(this.f5463a), yd.this.D);
                    return;
                } else if (Objects.equals(this.f5468f, "trigger")) {
                    yd.this.D.pause();
                    yd.this.D.removeAllUpdateListeners();
                    yd.this.D.removeAllListeners();
                    yd.this.D.end();
                    yd.this.D.cancel();
                    ve veVar = yd.this.N.get(Integer.valueOf(this.f5463a));
                    Objects.requireNonNull(veVar);
                    if (Objects.equals(veVar.A, "REPEAT")) {
                        yd ydVar2 = yd.this;
                        Node node = ydVar2.H.get(Integer.valueOf(this.f5463a));
                        Objects.requireNonNull(node);
                        RenderableInstance renderableInstance = node.getRenderableInstance();
                        ve veVar2 = yd.this.N.get(Integer.valueOf(this.f5463a));
                        Objects.requireNonNull(veVar2);
                        int i = veVar2.H;
                        ve veVar3 = yd.this.N.get(Integer.valueOf(this.f5463a));
                        Objects.requireNonNull(veVar3);
                        String str = veVar3.z;
                        ve veVar4 = yd.this.N.get(Integer.valueOf(this.f5463a));
                        Objects.requireNonNull(veVar4);
                        ydVar2.e(renderableInstance, "REPEAT", i, str, veVar4.j, "model");
                        return;
                    }
                    ve veVar5 = yd.this.N.get(Integer.valueOf(this.f5463a));
                    Objects.requireNonNull(veVar5);
                    if (Objects.equals(veVar5.A, "REPEAT_ONCE")) {
                        yd ydVar3 = yd.this;
                        Node node2 = ydVar3.H.get(Integer.valueOf(this.f5463a));
                        Objects.requireNonNull(node2);
                        RenderableInstance renderableInstance2 = node2.getRenderableInstance();
                        ve veVar6 = yd.this.N.get(Integer.valueOf(this.f5463a));
                        Objects.requireNonNull(veVar6);
                        int i2 = veVar6.H;
                        ve veVar7 = yd.this.N.get(Integer.valueOf(this.f5463a));
                        Objects.requireNonNull(veVar7);
                        String str2 = veVar7.z;
                        ve veVar8 = yd.this.N.get(Integer.valueOf(this.f5463a));
                        Objects.requireNonNull(veVar8);
                        ydVar3.e(renderableInstance2, "REPEAT_ONCE", i2, str2, veVar8.j, "model");
                        return;
                    }
                    return;
                } else {
                    yd.this.D.pause();
                    yd.this.D.removeAllListeners();
                    yd.this.D.setPropertyName("Idle");
                    yd.this.E.put(Integer.valueOf(this.f5463a), yd.this.D);
                    return;
                }
            }
            yd.this.D = this.f5467e.animate(iArr[0]);
            yd.this.D.setRepeatCount(0);
            yd.this.D.addListener(this);
            yd.this.D.start();
            yd.this.E.put(Integer.valueOf(this.f5463a), yd.this.D);
        }
    }

    /* compiled from: LoaderARContentSceneformARCore.java */
    /* loaded from: classes2.dex */
    public class c extends AnimatorListenerAdapter {

        /* renamed from: a  reason: collision with root package name */
        public final /* synthetic */ int f5470a;

        public c(int i) {
            this.f5470a = i;
        }

        @Override // android.animation.AnimatorListenerAdapter, android.animation.Animator.AnimatorListener
        public void onAnimationEnd(Animator animator) {
            super.onAnimationEnd(animator);
            ve veVar = yd.this.N.get(Integer.valueOf(this.f5470a));
            Objects.requireNonNull(veVar);
            if (Objects.equals(veVar.A, "REPEAT")) {
                yd ydVar = yd.this;
                Node node = ydVar.H.get(Integer.valueOf(this.f5470a));
                Objects.requireNonNull(node);
                RenderableInstance renderableInstance = node.getRenderableInstance();
                ve veVar2 = yd.this.N.get(Integer.valueOf(this.f5470a));
                Objects.requireNonNull(veVar2);
                int i = veVar2.H;
                ve veVar3 = yd.this.N.get(Integer.valueOf(this.f5470a));
                Objects.requireNonNull(veVar3);
                String str = veVar3.z;
                ve veVar4 = yd.this.N.get(Integer.valueOf(this.f5470a));
                Objects.requireNonNull(veVar4);
                ydVar.e(renderableInstance, "REPEAT", i, str, veVar4.j, "model");
                return;
            }
            ve veVar5 = yd.this.N.get(Integer.valueOf(this.f5470a));
            Objects.requireNonNull(veVar5);
            if (Objects.equals(veVar5.A, "REPEAT_ONCE")) {
                yd ydVar2 = yd.this;
                Node node2 = ydVar2.H.get(Integer.valueOf(this.f5470a));
                Objects.requireNonNull(node2);
                RenderableInstance renderableInstance2 = node2.getRenderableInstance();
                ve veVar6 = yd.this.N.get(Integer.valueOf(this.f5470a));
                Objects.requireNonNull(veVar6);
                int i2 = veVar6.H;
                ve veVar7 = yd.this.N.get(Integer.valueOf(this.f5470a));
                Objects.requireNonNull(veVar7);
                String str2 = veVar7.z;
                ve veVar8 = yd.this.N.get(Integer.valueOf(this.f5470a));
                Objects.requireNonNull(veVar8);
                ydVar2.e(renderableInstance2, "REPEAT_ONCE", i2, str2, veVar8.j, "model");
            }
        }
    }

    /* compiled from: LoaderARContentSceneformARCore.java */
    /* loaded from: classes2.dex */
    public class d extends AnimatorListenerAdapter {

        /* renamed from: a  reason: collision with root package name */
        public final /* synthetic */ int f5472a;

        /* renamed from: b  reason: collision with root package name */
        public final /* synthetic */ Node f5473b;

        public d(int i, Node node) {
            this.f5472a = i;
            this.f5473b = node;
        }

        @Override // android.animation.AnimatorListenerAdapter, android.animation.Animator.AnimatorListener
        public void onAnimationEnd(Animator animator) {
            super.onAnimationEnd(animator);
            if (this.f5472a == 1) {
                yd.this.s(this.f5473b);
            }
        }
    }

    /* compiled from: LoaderARContentSceneformARCore.java */
    /* loaded from: classes2.dex */
    public interface e {
        void a(String str);
    }

    @SuppressLint({"HardwareIds"})
    public yd(String str, String str2, String str3, Node node, ArFragment arFragment, Context context, Activity activity) {
        this.y = str;
        this.B = str2;
        this.z = str3;
        this.f5453d = node;
        this.f5456g = arFragment;
        this.f5450a = context;
        this.f5451b = activity;
        this.A = Settings.Secure.getString(context.getContentResolver(), "android_id");
        this.f5457h = new c.e.b.p000if.e(context);
    }

    public void a(ve veVar, MediaPlayer mediaPlayer, Node node) {
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
                m();
                return;
            case 1:
                NavigableMap navigableMap = this.O.get(Integer.valueOf(veVar.H));
                Iterator it = navigableMap.entrySet().iterator();
                String str2 = "";
                while (true) {
                    if (it.hasNext()) {
                        Map.Entry entry = (Map.Entry) it.next();
                        Map.Entry higherEntry = navigableMap.higherEntry((Integer) entry.getKey());
                        if (Objects.equals(this.M.get(Integer.valueOf(veVar.H)), "default")) {
                            str2 = (String) entry.getValue();
                        } else if (Objects.equals(entry.getValue(), this.M.get(Integer.valueOf(veVar.H)))) {
                            str2 = (String) higherEntry.getValue();
                        }
                    }
                }
                o(this.L.get(Integer.valueOf(veVar.H)), node, str2, veVar.H);
                return;
            case 2:
                n(veVar.f5356g);
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
                        e(node.getRenderableInstance(), veVar.A, veVar.H, veVar.z, true, "model");
                        return;
                    }
                }
                e(node.getRenderableInstance(), veVar.A, veVar.H, veVar.z, true, "model");
                return;
            case 4:
                try {
                    if (mediaPlayer.isPlaying()) {
                        mediaPlayer.pause();
                    } else {
                        mediaPlayer.start();
                    }
                    return;
                } catch (IllegalStateException e2) {
                    e2.printStackTrace();
                    return;
                }
            case 5:
                i(veVar.f5356g);
                return;
            case 6:
                v(veVar.f5356g);
                return;
            default:
                return;
        }
    }

    public final void b(int i) {
        this.p.postDelayed(this.q, i);
        if (this.w) {
            return;
        }
        this.w = true;
        this.f5456g.getArSceneView().getScene().addOnPeekTouchListener(new Scene.OnPeekTouchListener() { // from class: c.e.b.w9
            @Override // com.google.ar.sceneform.Scene.OnPeekTouchListener
            public final void onPeekTouch(HitTestResult hitTestResult, MotionEvent motionEvent) {
                yd ydVar = yd.this;
                if (ydVar.x) {
                    return;
                }
                if (!ydVar.v) {
                    ydVar.x(true);
                } else if (motionEvent.getAction() == 1) {
                    ydVar.p.postDelayed(ydVar.q, 2000L);
                }
            }
        });
    }

    public void c(ve veVar) {
        float f2 = veVar.f5350a;
        int i = veVar.f5351b;
        if (i == 0) {
            Node sunlight = this.f5456g.getArSceneView().getScene().getSunlight();
            Objects.requireNonNull(sunlight);
            Light light = sunlight.getLight();
            Objects.requireNonNull(light);
            light.setIntensity(200.0f * f2);
            StringBuilder sb = new StringBuilder();
            sb.append("Light0 ");
            sb.append(f2);
            sb.append(" ");
            Light light2 = this.f5456g.getArSceneView().getScene().getSunlight().getLight();
            Objects.requireNonNull(light2);
            sb.append(light2.getIntensity());
            Log.d("LoaderARContentSceneformARCore", sb.toString());
        } else if (i == 1) {
            Light build = Light.builder(Light.Type.DIRECTIONAL).setColor(new Color(1.0f, 1.0f, 1.0f)).setShadowCastingEnabled(true).setIntensity(50.0f * f2).build();
            this.f5456g.getArSceneView().getScene().getSunlight().setEnabled(false);
            this.f5456g.getArSceneView().getScene().setLightEstimate(new Color(1.0f, 1.0f, 1.0f), f2 / 5.0f);
            Node node = new Node();
            node.setWorldPosition(new Vector3(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 10.0f, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD));
            node.setLight(build);
            node.setWorldRotation(Quaternion.eulerAngles(new Vector3(-30.0f, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 30.0f)));
            node.setParent(this.f5453d);
            this.t++;
            StringBuilder x = c.b.a.a.a.x("Light1 ");
            x.append(build.getIntensity());
            x.append(" ");
            x.append(this.f5456g.getArSceneView().getScene().getSunlight().getLight().getIntensity());
            Log.d("LoaderARContentSceneformARCore", x.toString());
        } else if (i == 2) {
            Light.Type type = Light.Type.DIRECTIONAL;
            Light build2 = Light.builder(type).setColor(new Color(1.0f, 1.0f, 1.0f)).setShadowCastingEnabled(false).setIntensity(500.0f).build();
            Light build3 = Light.builder(type).setColor(new Color(1.0f, 1.0f, 1.0f)).setShadowCastingEnabled(false).setIntensity(500.0f).build();
            this.f5456g.getArSceneView().getScene().getSunlight().setLight(build2);
            this.f5456g.getArSceneView().getScene().setLightEstimate(new Color(1.0f, 1.0f, 1.0f), f2 / 5.0f);
            Node node2 = new Node();
            node2.setWorldPosition(new Vector3(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 10.0f, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD));
            node2.setWorldRotation(Quaternion.eulerAngles(new Vector3(-30.0f, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 30.0f)));
            node2.setLight(build3);
            node2.setParent(this.f5453d);
            this.t++;
            StringBuilder x2 = c.b.a.a.a.x("Light ");
            x2.append(build2.getIntensity());
            x2.append(" ");
            x2.append(this.f5456g.getArSceneView().getScene().getSunlight().getLight().getIntensity());
            Log.d("LoaderARContentSceneformARCore", x2.toString());
        }
    }

    public void d(final Node node, ve veVar, int i) {
        String str;
        Vector3 vector3;
        Vector3 vector32;
        Vector3 vector33;
        Vector3 localPosition;
        Vector3 vector34;
        Vector3 localPosition2;
        Vector3 vector35;
        Vector3 localScale;
        Vector3 vector36;
        Vector3 localPosition3;
        Vector3 vector37;
        Vector3 localPosition4;
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
            objectAnimator.addListener(new d(i, node));
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
                        localPosition = new Vector3(1.0f, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
                    } else {
                        vector33 = new Vector3(-1.0f, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
                        localPosition = node.getLocalPosition();
                    }
                    objectAnimator.setObjectValues(vector33, localPosition);
                    objectAnimator.setPropertyName("localPosition");
                    objectAnimator.setEvaluator(new Vector3Evaluator());
                    break;
                case 2:
                    if (i == 1) {
                        vector34 = node.getLocalPosition();
                        localPosition2 = new Vector3(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 1.0f, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
                    } else {
                        vector34 = new Vector3(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, -1.0f, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
                        localPosition2 = node.getLocalPosition();
                    }
                    objectAnimator.setObjectValues(vector34, localPosition2);
                    objectAnimator.setPropertyName("localPosition");
                    objectAnimator.setEvaluator(new Vector3Evaluator());
                    break;
                case 3:
                    objectAnimator.setFloatValues(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 1.0f);
                    objectAnimator.setPropertyName("alpha");
                    break;
                case 4:
                    if (i == 1) {
                        vector35 = new Vector3(1.0f, 1.0f, 1.0f);
                        localScale = new Vector3(2.0f, 2.0f, 2.0f);
                    } else {
                        vector35 = new Vector3(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
                        localScale = node.getLocalScale();
                    }
                    objectAnimator.setObjectValues(vector35, localScale);
                    objectAnimator.setPropertyName("localScale");
                    objectAnimator.setEvaluator(new Vector3Evaluator());
                    break;
                case 5:
                    objectAnimator.setFloatValues(1.0f, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
                    objectAnimator.setPropertyName("alpha");
                    break;
                case 6:
                    if (i == 1) {
                        vector36 = node.getLocalPosition();
                        localPosition3 = new Vector3(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, -1.0f, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
                    } else {
                        vector36 = new Vector3(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 1.0f, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
                        localPosition3 = node.getLocalPosition();
                    }
                    objectAnimator.setObjectValues(vector36, localPosition3);
                    objectAnimator.setPropertyName("localPosition");
                    objectAnimator.setEvaluator(new Vector3Evaluator());
                    break;
                case 7:
                    if (i == 1) {
                        vector37 = node.getLocalPosition();
                        localPosition4 = new Vector3(-1.0f, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
                    } else {
                        vector37 = new Vector3(1.0f, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
                        localPosition4 = node.getLocalPosition();
                    }
                    objectAnimator.setObjectValues(vector37, localPosition4);
                    objectAnimator.setPropertyName("localPosition");
                    objectAnimator.setEvaluator(new Vector3Evaluator());
                    break;
                case '\b':
                    objectAnimator.setFloatValues(2.0f, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
                    objectAnimator.setInterpolator(new BounceInterpolator());
                    objectAnimator.addUpdateListener(new ValueAnimator.AnimatorUpdateListener() { // from class: c.e.b.r8
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

    public final void e(RenderableInstance renderableInstance, String str, int i, String str2, boolean z, String str3) {
        this.G.put(Integer.valueOf(i), str3);
        this.F.put(Integer.valueOf(i), new String[]{str2, str});
        if (str2.equals("IDLE") || str2.length() == 0 || renderableInstance.getFilamentAsset().getAnimator().getAnimationCount() == 0) {
            return;
        }
        int animationCount = renderableInstance.getAnimationCount();
        this.D = renderableInstance.animate(0);
        if (str.equals("REPEAT") && animationCount == 0) {
            this.D.setRepeatCount(-1);
        } else {
            this.D.setRepeatCount(0);
        }
        if (z && str2.equals("ALL")) {
            this.D.start();
        } else {
            this.D.pause();
        }
        if (str2.equals("ALL")) {
            this.D.addListener(new b(i, new int[]{0}, animationCount, str, renderableInstance, str3));
        } else {
            this.D = renderableInstance.animate(str2);
            if (str.equals("REPEAT") && animationCount > 0) {
                this.D.setRepeatCount(-1);
            } else {
                this.D.setRepeatCount(0);
            }
            if (z) {
                this.D.start();
            } else {
                this.D.pause();
            }
            if (Objects.equals(str3, "trigger")) {
                this.D.addListener(new c(i));
            }
        }
        this.E.put(Integer.valueOf(i), this.D);
    }

    public void f(ve veVar, MediaPlayer mediaPlayer) {
        try {
            mediaPlayer.setDataSource(veVar.f5356g);
            mediaPlayer.prepare();
        } catch (Exception e2) {
            e2.printStackTrace();
        }
    }

    /* JADX WARN: Unsupported multi-entry loop pattern (BACK_EDGE: B:19:0x00c0 -> B:25:0x00ca). Please submit an issue!!! */
    /* JADX WARN: Unsupported multi-entry loop pattern (BACK_EDGE: B:20:0x00c2 -> B:25:0x00ca). Please submit an issue!!! */
    public final void g(JSONObject jSONObject) {
        try {
            int i = jSONObject.getInt("contentTypeId");
            try {
                Integer.parseInt(jSONObject.getString("id"));
                final ve veVar = new ve(jSONObject, i);
                this.O.put(Integer.valueOf(veVar.H), veVar.T);
                switch (i) {
                    case 1:
                        this.f5451b.runOnUiThread(new Runnable() { // from class: c.e.b.k9
                            @Override // java.lang.Runnable
                            public final void run() {
                                final yd ydVar = yd.this;
                                final ve veVar2 = veVar;
                                Objects.requireNonNull(ydVar);
                                float f2 = veVar2.m[1];
                                final SimpleTransformableNode simpleTransformableNode = new SimpleTransformableNode(ydVar.f5456g.getTransformationSystem());
                                simpleTransformableNode.setParent(ydVar.f5453d);
                                float[] fArr = veVar2.l;
                                final Node Q = c.b.a.a.a.Q(simpleTransformableNode, new Vector3(fArr[0], fArr[1], fArr[2]), simpleTransformableNode);
                                float[] fArr2 = veVar2.q;
                                Q.setLocalRotation(new Quaternion(-fArr2[0], -fArr2[1], fArr2[2], fArr2[3]));
                                float[] fArr3 = veVar2.n;
                                if (fArr3[0] != StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                                    Q.setLocalScale(new Vector3(fArr3[0], fArr3[1], fArr3[2]).scaled(veVar2.p));
                                }
                                ydVar.f5456g.getArSceneView().getScene().addOnPeekTouchListener(new Scene.OnPeekTouchListener() { // from class: c.e.b.v9
                                    @Override // com.google.ar.sceneform.Scene.OnPeekTouchListener
                                    public final void onPeekTouch(HitTestResult hitTestResult, MotionEvent motionEvent) {
                                        yd.this.f5456g.getTransformationSystem().onTouch(hitTestResult, motionEvent);
                                    }
                                });
                                c.b.a.a.a.c(f2, 0.01f, simpleTransformableNode.getScaleController(), simpleTransformableNode).setMaxScale(f2 * 0.08f);
                                final MediaPlayer l = ydVar.l();
                                l.setOnCompletionListener(new MediaPlayer.OnCompletionListener() { // from class: c.e.b.qa
                                    @Override // android.media.MediaPlayer.OnCompletionListener
                                    public final void onCompletion(MediaPlayer mediaPlayer) {
                                        yd ydVar2 = yd.this;
                                        ve veVar3 = veVar2;
                                        Objects.requireNonNull(ydVar2);
                                        if (Objects.equals(veVar3.f5355f, "GOTO_SCENE_AFTER_CONTENT_END")) {
                                            ydVar2.C = veVar3.f5356g;
                                            ydVar2.m();
                                        }
                                    }
                                });
                                final ExternalTexture externalTexture = new ExternalTexture();
                                l.setSurface(externalTexture.getSurface());
                                l.setAudioStreamType(3);
                                try {
                                    l.setScreenOnWhilePlaying(true);
                                    l.setDataSource(veVar2.f5354e);
                                    l.prepareAsync();
                                    l.setOnPreparedListener(new MediaPlayer.OnPreparedListener() { // from class: c.e.b.oa
                                        @Override // android.media.MediaPlayer.OnPreparedListener
                                        public final void onPrepared(MediaPlayer mediaPlayer) {
                                            final yd ydVar2 = yd.this;
                                            final ve veVar3 = veVar2;
                                            final MediaPlayer mediaPlayer2 = l;
                                            final ExternalTexture externalTexture2 = externalTexture;
                                            final Node node = Q;
                                            final SimpleTransformableNode simpleTransformableNode2 = simpleTransformableNode;
                                            Objects.requireNonNull(ydVar2);
                                            final float videoHeight = mediaPlayer.getVideoHeight();
                                            final float videoWidth = mediaPlayer.getVideoWidth();
                                            int i2 = veVar3.u ? R.raw.chroma_key_video_material : R.raw.augmented_video_material;
                                            if (veVar3.o) {
                                                mediaPlayer2.setLooping(true);
                                            }
                                            Material.builder().setSource(ydVar2.f5451b, i2).build().thenAccept(new Consumer() { // from class: c.e.b.v8
                                                /* JADX DEBUG: Multi-variable search result rejected for r1v0, resolved type: c.e.b.yd */
                                                /* JADX DEBUG: Multi-variable search result rejected for r4v3, resolved type: com.google.ar.sceneform.rendering.ModelRenderable */
                                                /* JADX WARN: Multi-variable type inference failed */
                                                /* JADX WARN: Type inference failed for: r0v1 */
                                                /* JADX WARN: Type inference failed for: r0v2, types: [int, boolean] */
                                                /* JADX WARN: Type inference failed for: r0v6 */
                                                @Override // java.util.function.Consumer
                                                public final void accept(Object obj) {
                                                    ?? r0;
                                                    final yd ydVar3 = yd.this;
                                                    final ve veVar4 = veVar3;
                                                    float f3 = videoWidth;
                                                    float f4 = videoHeight;
                                                    ExternalTexture externalTexture3 = externalTexture2;
                                                    Node node2 = node;
                                                    SimpleTransformableNode simpleTransformableNode3 = simpleTransformableNode2;
                                                    final MediaPlayer mediaPlayer3 = mediaPlayer2;
                                                    Material material = (Material) obj;
                                                    Objects.requireNonNull(ydVar3);
                                                    if (veVar4.u) {
                                                        float[] fArr4 = veVar4.y;
                                                        r0 = 0;
                                                        material.setFloat4("keyColor", fArr4[0], fArr4[1], fArr4[2], 1.0f);
                                                    } else {
                                                        r0 = 0;
                                                    }
                                                    float f5 = f3 / f4;
                                                    Renderable makeCube = ShapeFactory.makeCube(new Vector3(f5 * 4.4f, 4.4f, 1.0E-4f), Vector3.zero(), material);
                                                    makeCube.setShadowCaster(r0);
                                                    makeCube.setShadowReceiver(r0);
                                                    makeCube.getMaterial().setExternalTexture("videoTexture", externalTexture3);
                                                    node2.setRenderable(makeCube);
                                                    final Node node3 = new Node();
                                                    node3.setName("playPauseButton");
                                                    ydVar3.o.add(node3);
                                                    node3.setParent(simpleTransformableNode3);
                                                    node3.setLocalPosition(new Vector3(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 0.001f));
                                                    node3.setOnAccurateTapListner(new Node.OnAccurateTapListner() { // from class: c.e.b.z7
                                                        @Override // com.google.ar.sceneform.Node.OnAccurateTapListner
                                                        public final void onAccurateTap(HitTestResult hitTestResult, MotionEvent motionEvent) {
                                                            Node node4 = Node.this;
                                                            MediaPlayer mediaPlayer4 = mediaPlayer3;
                                                            ImageView imageView = (ImageView) ((ViewRenderable) node4.getRenderable()).getView().findViewById(R.id.img_loader_view);
                                                            if (mediaPlayer4.isPlaying()) {
                                                                mediaPlayer4.pause();
                                                                imageView.setImageResource(R.drawable.play);
                                                                return;
                                                            }
                                                            mediaPlayer4.start();
                                                            imageView.setImageResource(R.drawable.pause);
                                                        }
                                                    });
                                                    simpleTransformableNode3.setOnAccurateTapListner(new Node.OnAccurateTapListner() { // from class: c.e.b.c9
                                                        @Override // com.google.ar.sceneform.Node.OnAccurateTapListner
                                                        public final void onAccurateTap(HitTestResult hitTestResult, MotionEvent motionEvent) {
                                                            yd ydVar4 = yd.this;
                                                            ve veVar5 = veVar4;
                                                            MediaPlayer mediaPlayer4 = mediaPlayer3;
                                                            Node node4 = node3;
                                                            Objects.requireNonNull(ydVar4);
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
                                                                case 53:
                                                                    if (str.equals("5")) {
                                                                        c2 = 3;
                                                                        break;
                                                                    }
                                                                    break;
                                                                case 827942371:
                                                                    if (str.equals("GOTO_SCENE_AFTER_CONTENT_END")) {
                                                                        c2 = 4;
                                                                        break;
                                                                    }
                                                                    break;
                                                            }
                                                            switch (c2) {
                                                                case 0:
                                                                case 3:
                                                                    ydVar4.C = veVar5.f5356g;
                                                                    ydVar4.m();
                                                                    return;
                                                                case 1:
                                                                    Intent intent = new Intent(ydVar4.f5450a, VideoActivity.class);
                                                                    intent.putExtra("videoUrl", veVar5.f5354e);
                                                                    intent.putExtra("loop", veVar5.o);
                                                                    intent.putExtra("currenttime", mediaPlayer4.getCurrentPosition());
                                                                    ydVar4.f5450a.startActivity(intent);
                                                                    return;
                                                                case 2:
                                                                case 4:
                                                                    ImageView imageView = (ImageView) ((ViewRenderable) node4.getRenderable()).getView().findViewById(R.id.img_loader_view);
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
                                                    ViewRenderable.builder().setView(ydVar3.f5451b, R.layout.img_view).build().thenAccept(new Consumer() { // from class: c.e.b.ua
                                                        @Override // java.util.function.Consumer
                                                        public final void accept(Object obj2) {
                                                            ve veVar5 = ve.this;
                                                            Node node4 = node3;
                                                            ViewRenderable viewRenderable = (ViewRenderable) obj2;
                                                            viewRenderable.setHorizontalAlignment(ViewRenderable.HorizontalAlignment.CENTER);
                                                            viewRenderable.setVerticalAlignment(ViewRenderable.VerticalAlignment.CENTER);
                                                            ImageView imageView = (ImageView) viewRenderable.getView().findViewById(R.id.img_loader_view);
                                                            if (veVar5.j) {
                                                                imageView.setImageResource(R.drawable.pause);
                                                            } else {
                                                                imageView.setImageResource(R.drawable.play);
                                                            }
                                                            node4.setRenderable(viewRenderable);
                                                            node4.setLocalScale(new Vector3(1.0f, 1.0f, 1.0f));
                                                        }
                                                    }).exceptionally(new Function() { // from class: c.e.b.p8
                                                        @Override // java.util.function.Function
                                                        public final Object apply(Object obj2) {
                                                            yd ydVar4 = yd.this;
                                                            Throwable th = (Throwable) obj2;
                                                            int i3 = ydVar4.s - 1;
                                                            ydVar4.s = i3;
                                                            if (i3 == 0) {
                                                                ydVar4.j();
                                                            }
                                                            Log.e("LoaderARContentSceneformARCore", "Unable to load  renderable");
                                                            return null;
                                                        }
                                                    });
                                                    final Node node4 = new Node();
                                                    ydVar3.o.add(node4);
                                                    node4.setParent(node2);
                                                    node4.setLocalPosition(new Vector3(f5 * 2.0f, -2.0f, 0.001f));
                                                    node4.setOnAccurateTapListner(new Node.OnAccurateTapListner() { // from class: c.e.b.o7
                                                        @Override // com.google.ar.sceneform.Node.OnAccurateTapListner
                                                        public final void onAccurateTap(HitTestResult hitTestResult, MotionEvent motionEvent) {
                                                            yd ydVar4 = yd.this;
                                                            ve veVar5 = veVar4;
                                                            Objects.requireNonNull(ydVar4);
                                                            Intent intent = new Intent(ydVar4.f5450a, VideoActivity.class);
                                                            intent.putExtra("videoUrl", veVar5.f5354e);
                                                            intent.putExtra("loop", veVar5.o);
                                                            ydVar4.f5450a.startActivity(intent);
                                                        }
                                                    });
                                                    ViewRenderable.builder().setView(ydVar3.f5451b, R.layout.img_view).build().thenAccept(new Consumer() { // from class: c.e.b.t9
                                                        @Override // java.util.function.Consumer
                                                        public final void accept(Object obj2) {
                                                            Node node5 = Node.this;
                                                            ViewRenderable viewRenderable = (ViewRenderable) obj2;
                                                            ((ImageView) viewRenderable.getView().findViewById(R.id.img_loader_view)).setImageResource(R.drawable.fullscreen);
                                                            node5.setRenderable(viewRenderable);
                                                            c.b.a.a.a.C(0.6f, 0.6f, 0.6f, node5);
                                                        }
                                                    }).exceptionally(new Function() { // from class: c.e.b.m9
                                                        @Override // java.util.function.Function
                                                        public final Object apply(Object obj2) {
                                                            yd ydVar4 = yd.this;
                                                            Throwable th = (Throwable) obj2;
                                                            int i3 = ydVar4.s - 1;
                                                            ydVar4.s = i3;
                                                            if (i3 == 0) {
                                                                ydVar4.j();
                                                            }
                                                            Log.e("LoaderARContentSceneformARCore", "Unable to load  renderable");
                                                            return null;
                                                        }
                                                    });
                                                    if (veVar4.j) {
                                                        ydVar3.I.put(mediaPlayer3, Boolean.TRUE);
                                                        ydVar3.b(10);
                                                    } else {
                                                        ydVar3.b(2000);
                                                    }
                                                    mediaPlayer3.seekTo(1);
                                                    ydVar3.d(node2, veVar4, r0);
                                                    ydVar3.K.put(node2, veVar4);
                                                    int i3 = ydVar3.s - 1;
                                                    ydVar3.s = i3;
                                                    if (i3 == 0) {
                                                        ydVar3.j();
                                                    }
                                                }
                                            }).exceptionally(new Function() { // from class: c.e.b.m8
                                                @Override // java.util.function.Function
                                                public final Object apply(Object obj) {
                                                    yd ydVar3 = yd.this;
                                                    Throwable th = (Throwable) obj;
                                                    ydVar3.f5453d.removeChild(simpleTransformableNode2);
                                                    int i3 = ydVar3.s - 1;
                                                    ydVar3.s = i3;
                                                    if (i3 == 0) {
                                                        ydVar3.j();
                                                    }
                                                    Log.e("LoaderARContentSceneformARCore", "Unable to load  renderable");
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
                        this.f5451b.runOnUiThread(new Runnable() { // from class: c.e.b.ta
                            @Override // java.lang.Runnable
                            public final void run() {
                                MediaPlayer mediaPlayer;
                                final yd ydVar = yd.this;
                                final ve veVar2 = veVar;
                                Objects.requireNonNull(ydVar);
                                float f2 = veVar2.m[1];
                                final String a2 = new cf().a(veVar2.f5354e);
                                final String r = c.b.a.a.a.r("https://img.youtube.com/vi/", a2, "/hqdefault.jpg");
                                SimpleTransformableNode simpleTransformableNode = new SimpleTransformableNode(ydVar.f5456g.getTransformationSystem());
                                simpleTransformableNode.setParent(ydVar.f5453d);
                                float[] fArr = veVar2.l;
                                final Node Q = c.b.a.a.a.Q(simpleTransformableNode, new Vector3(fArr[0], fArr[1], fArr[2]), simpleTransformableNode);
                                float[] fArr2 = veVar2.q;
                                Q.setLocalRotation(new Quaternion(-fArr2[0], -fArr2[1], fArr2[2], fArr2[3]));
                                float[] fArr3 = veVar2.n;
                                if (fArr3[0] != StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                                    Q.setLocalScale(new Vector3(fArr3[0], fArr3[1], fArr3[2]).scaled(veVar2.p));
                                }
                                ydVar.f5456g.getArSceneView().getScene().addOnPeekTouchListener(new Scene.OnPeekTouchListener() { // from class: c.e.b.ea
                                    @Override // com.google.ar.sceneform.Scene.OnPeekTouchListener
                                    public final void onPeekTouch(HitTestResult hitTestResult, MotionEvent motionEvent) {
                                        yd.this.f5456g.getTransformationSystem().onTouch(hitTestResult, motionEvent);
                                    }
                                });
                                c.b.a.a.a.S(f2, 0.5f, c.b.a.a.a.c(f2, 0.05f, simpleTransformableNode.getScaleController(), simpleTransformableNode)).setView(ydVar.f5451b, R.layout.img_view).build().thenAccept(new Consumer() { // from class: c.e.b.ia
                                    @Override // java.util.function.Consumer
                                    public final void accept(Object obj) {
                                        yd ydVar2 = yd.this;
                                        String str = r;
                                        Node node = Q;
                                        ViewRenderable viewRenderable = (ViewRenderable) obj;
                                        Objects.requireNonNull(ydVar2);
                                        c.c.a.b.d(ydVar2.f5451b).k(str).B((ImageView) viewRenderable.getView().findViewById(R.id.img_loader_view));
                                        viewRenderable.setHorizontalAlignment(ViewRenderable.HorizontalAlignment.CENTER);
                                        viewRenderable.setVerticalAlignment(ViewRenderable.VerticalAlignment.CENTER);
                                        node.setRenderable(viewRenderable);
                                    }
                                }).exceptionally(new Function() { // from class: c.e.b.o8
                                    @Override // java.util.function.Function
                                    public final Object apply(Object obj) {
                                        yd ydVar2 = yd.this;
                                        Throwable th = (Throwable) obj;
                                        ydVar2.f5453d.removeChild(Q);
                                        int i2 = ydVar2.s - 1;
                                        ydVar2.s = i2;
                                        if (i2 == 0) {
                                            ydVar2.j();
                                        }
                                        Log.e("LoaderARContentSceneformARCore", "Unable to load  renderable");
                                        return null;
                                    }
                                });
                                final Node node = new Node();
                                node.setParent(Q);
                                node.setLocalPosition(new Vector3(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 0.01f));
                                if (veVar2.f5355f.equals("PLAY_SOUND")) {
                                    mediaPlayer = ydVar.l();
                                    mediaPlayer.setLooping(veVar2.i);
                                    ydVar.f(veVar2, mediaPlayer);
                                    if (veVar2.f5357h) {
                                        ydVar.I.put(mediaPlayer, Boolean.TRUE);
                                    }
                                } else {
                                    mediaPlayer = null;
                                }
                                final MediaPlayer mediaPlayer2 = mediaPlayer;
                                Q.setOnAccurateTapListner(new Node.OnAccurateTapListner() { // from class: c.e.b.ma
                                    @Override // com.google.ar.sceneform.Node.OnAccurateTapListner
                                    public final void onAccurateTap(HitTestResult hitTestResult, MotionEvent motionEvent) {
                                        yd ydVar2 = yd.this;
                                        ve veVar3 = veVar2;
                                        String str = a2;
                                        MediaPlayer mediaPlayer3 = mediaPlayer2;
                                        Node node2 = Q;
                                        Objects.requireNonNull(ydVar2);
                                        if (Objects.equals(veVar3.f5355f, "NO_ACTION")) {
                                            ydVar2.q();
                                            Intent intent = new Intent(ydVar2.f5450a, YoutubeView.class);
                                            intent.putExtra("youtubeID", str);
                                            ydVar2.f5450a.startActivity(intent);
                                            return;
                                        }
                                        ydVar2.a(veVar3, mediaPlayer3, node2);
                                    }
                                });
                                ViewRenderable.builder().setView(ydVar.f5451b, R.layout.img_view).build().thenAccept(new Consumer() { // from class: c.e.b.r7
                                    @Override // java.util.function.Consumer
                                    public final void accept(Object obj) {
                                        yd ydVar2 = yd.this;
                                        Node node2 = node;
                                        Node node3 = Q;
                                        ve veVar3 = veVar2;
                                        ViewRenderable viewRenderable = (ViewRenderable) obj;
                                        Objects.requireNonNull(ydVar2);
                                        ((ImageView) viewRenderable.getView().findViewById(R.id.img_loader_view)).setImageResource(R.drawable.play_youtube);
                                        viewRenderable.setHorizontalAlignment(ViewRenderable.HorizontalAlignment.CENTER);
                                        viewRenderable.setVerticalAlignment(ViewRenderable.VerticalAlignment.CENTER);
                                        node2.setRenderable(viewRenderable);
                                        c.b.a.a.a.C(0.2f, 0.2f, 0.2f, node2);
                                        ydVar2.d(node3, veVar3, 0);
                                        ydVar2.K.put(node3, veVar3);
                                        int i2 = ydVar2.s - 1;
                                        ydVar2.s = i2;
                                        if (i2 == 0) {
                                            ydVar2.j();
                                        }
                                    }
                                }).exceptionally(new Function() { // from class: c.e.b.h9
                                    @Override // java.util.function.Function
                                    public final Object apply(Object obj) {
                                        Throwable th = (Throwable) obj;
                                        Objects.requireNonNull(yd.this);
                                        Log.e("LoaderARContentSceneformARCore", "Unable to load  renderable");
                                        return null;
                                    }
                                });
                            }
                        });
                        break;
                    case 3:
                    case 5:
                        this.f5451b.runOnUiThread(new Runnable() { // from class: c.e.b.ka
                            @Override // java.lang.Runnable
                            public final void run() {
                                final yd ydVar = yd.this;
                                final ve veVar2 = veVar;
                                Objects.requireNonNull(ydVar);
                                float f2 = veVar2.m[1];
                                final SimpleTransformableNode simpleTransformableNode = new SimpleTransformableNode(ydVar.f5456g.getTransformationSystem());
                                simpleTransformableNode.setParent(ydVar.f5453d);
                                float[] fArr = veVar2.l;
                                final Node Q = c.b.a.a.a.Q(simpleTransformableNode, new Vector3(fArr[0], fArr[1], fArr[2]), simpleTransformableNode);
                                float[] fArr2 = veVar2.q;
                                Q.setLocalRotation(new Quaternion(-fArr2[0], -fArr2[1], fArr2[2], fArr2[3]));
                                float[] fArr3 = veVar2.n;
                                if (fArr3[0] != StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                                    Q.setLocalScale(new Vector3(fArr3[0], fArr3[1], fArr3[2]).scaled(veVar2.p));
                                }
                                ydVar.f5456g.getArSceneView().getScene().addOnPeekTouchListener(new Scene.OnPeekTouchListener() { // from class: c.e.b.fa
                                    @Override // com.google.ar.sceneform.Scene.OnPeekTouchListener
                                    public final void onPeekTouch(HitTestResult hitTestResult, MotionEvent motionEvent) {
                                        yd.this.f5456g.getTransformationSystem().onTouch(hitTestResult, motionEvent);
                                    }
                                });
                                c.b.a.a.a.S(f2, 0.7f, c.b.a.a.a.c(f2, 0.07f, simpleTransformableNode.getScaleController(), simpleTransformableNode)).setView(ydVar.f5451b, R.layout.image_view_threesixty).build().thenAccept(new Consumer() { // from class: c.e.b.e9
                                    @Override // java.util.function.Consumer
                                    public final void accept(Object obj) {
                                        yd ydVar2 = yd.this;
                                        Node node = Q;
                                        ve veVar3 = veVar2;
                                        ViewRenderable viewRenderable = (ViewRenderable) obj;
                                        Objects.requireNonNull(ydVar2);
                                        ((ImageView) viewRenderable.getView().findViewById(R.id.threesixty_img)).setImageResource(R.drawable.button360);
                                        viewRenderable.setHorizontalAlignment(ViewRenderable.HorizontalAlignment.CENTER);
                                        c.b.a.a.a.J(viewRenderable, ViewRenderable.VerticalAlignment.CENTER, false, false);
                                        node.setRenderable(viewRenderable);
                                        ydVar2.d(node, veVar3, 0);
                                        ydVar2.K.put(node, veVar3);
                                        int i2 = ydVar2.s - 1;
                                        ydVar2.s = i2;
                                        if (i2 == 0) {
                                            ydVar2.j();
                                        }
                                    }
                                }).exceptionally(new Function() { // from class: c.e.b.y7
                                    @Override // java.util.function.Function
                                    public final Object apply(Object obj) {
                                        yd ydVar2 = yd.this;
                                        Throwable th = (Throwable) obj;
                                        ydVar2.f5453d.removeChild(simpleTransformableNode);
                                        int i2 = ydVar2.s - 1;
                                        ydVar2.s = i2;
                                        if (i2 == 0) {
                                            ydVar2.j();
                                        }
                                        Log.e("LoaderARContentSceneformARCore", "Unable to load  renderable");
                                        return null;
                                    }
                                });
                                simpleTransformableNode.setOnAccurateTapListner(new Node.OnAccurateTapListner() { // from class: c.e.b.x9
                                    @Override // com.google.ar.sceneform.Node.OnAccurateTapListner
                                    public final void onAccurateTap(HitTestResult hitTestResult, MotionEvent motionEvent) {
                                        yd.this.r(veVar2);
                                    }
                                });
                                if (veVar2.k) {
                                    ydVar.r(veVar2);
                                }
                            }
                        });
                        break;
                    case 4:
                        this.f5451b.runOnUiThread(new Runnable() { // from class: c.e.b.b9
                            @Override // java.lang.Runnable
                            public final void run() {
                                final MediaPlayer mediaPlayer;
                                final yd ydVar = yd.this;
                                final ve veVar2 = veVar;
                                Objects.requireNonNull(ydVar);
                                final String[] split = veVar2.f5354e.split(",");
                                final int[] iArr = {0};
                                float f2 = veVar2.m[1];
                                final Node node = new Node();
                                final Node node2 = new Node();
                                final SimpleTransformableNode simpleTransformableNode = new SimpleTransformableNode(ydVar.f5456g.getTransformationSystem());
                                simpleTransformableNode.setParent(ydVar.f5453d);
                                float[] fArr = veVar2.l;
                                final Node Q = c.b.a.a.a.Q(simpleTransformableNode, new Vector3(fArr[0], fArr[1], fArr[2]), simpleTransformableNode);
                                float[] fArr2 = veVar2.q;
                                Q.setLocalRotation(new Quaternion(-fArr2[0], -fArr2[1], fArr2[2], fArr2[3]));
                                float[] fArr3 = veVar2.n;
                                if (fArr3[0] != StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                                    Q.setLocalScale(new Vector3(fArr3[0], fArr3[1], fArr3[2]).scaled(veVar2.p));
                                }
                                ydVar.f5456g.getArSceneView().getScene().addOnPeekTouchListener(new Scene.OnPeekTouchListener() { // from class: c.e.b.x8
                                    @Override // com.google.ar.sceneform.Scene.OnPeekTouchListener
                                    public final void onPeekTouch(HitTestResult hitTestResult, MotionEvent motionEvent) {
                                        yd.this.f5456g.getTransformationSystem().onTouch(hitTestResult, motionEvent);
                                    }
                                });
                                c.b.a.a.a.S(f2, 0.4f, c.b.a.a.a.c(f2, 0.04f, simpleTransformableNode.getScaleController(), simpleTransformableNode)).setView(ydVar.f5451b, R.layout.imageview_slideshow).build().thenAccept(new Consumer() { // from class: c.e.b.ja
                                    @Override // java.util.function.Consumer
                                    public final void accept(Object obj) {
                                        int i2;
                                        yd ydVar2 = yd.this;
                                        String[] strArr = split;
                                        Node node3 = node;
                                        Node node4 = node2;
                                        Node node5 = Q;
                                        ve veVar3 = veVar2;
                                        int[] iArr2 = iArr;
                                        ViewRenderable viewRenderable = (ViewRenderable) obj;
                                        Objects.requireNonNull(ydVar2);
                                        ImageView imageView = (ImageView) viewRenderable.getView().findViewById(R.id.img_loader_view);
                                        c.c.a.b.d(ydVar2.f5451b).k(strArr[0]).C(new td(ydVar2, node3, node4)).B(imageView);
                                        viewRenderable.setHorizontalAlignment(ViewRenderable.HorizontalAlignment.CENTER);
                                        viewRenderable.setVerticalAlignment(ViewRenderable.VerticalAlignment.CENTER);
                                        node5.setRenderable(viewRenderable);
                                        if (veVar3.j) {
                                            Timer timer = new Timer();
                                            timer.schedule(new vd(ydVar2, timer, iArr2, strArr, node3, node4, imageView), 0L, veVar3.O * 1000);
                                            i2 = 0;
                                        } else if (strArr.length > 1) {
                                            ydVar2.p(node5, 1, iArr2, strArr, imageView, node3, node4);
                                            ydVar2.p(node5, -1, iArr2, strArr, imageView, node3, node4);
                                            i2 = 0;
                                        } else {
                                            i2 = 0;
                                        }
                                        ydVar2.d(node5, veVar3, i2);
                                        ydVar2.K.put(node5, veVar3);
                                        int i3 = ydVar2.s - 1;
                                        ydVar2.s = i3;
                                        if (i3 == 0) {
                                            ydVar2.j();
                                        }
                                    }
                                }).exceptionally(new Function() { // from class: c.e.b.w8
                                    @Override // java.util.function.Function
                                    public final Object apply(Object obj) {
                                        yd ydVar2 = yd.this;
                                        Throwable th = (Throwable) obj;
                                        ydVar2.f5453d.removeChild(simpleTransformableNode);
                                        int i2 = ydVar2.s - 1;
                                        ydVar2.s = i2;
                                        if (i2 == 0) {
                                            ydVar2.j();
                                        }
                                        Log.e("LoaderARContentSceneformARCore", "Unable to load  renderable");
                                        return null;
                                    }
                                });
                                if (veVar2.f5355f.equals("PLAY_SOUND")) {
                                    mediaPlayer = ydVar.l();
                                    mediaPlayer.setLooping(veVar2.i);
                                    ydVar.f(veVar2, mediaPlayer);
                                    if (veVar2.f5357h) {
                                        ydVar.I.put(mediaPlayer, Boolean.TRUE);
                                    }
                                } else {
                                    mediaPlayer = null;
                                }
                                if (veVar2.f5355f.equals("PLAY_SOUND") && veVar2.f5357h) {
                                    ydVar.f(veVar2, mediaPlayer);
                                }
                                Q.setOnAccurateTapListner(new Node.OnAccurateTapListner() { // from class: c.e.b.s8
                                    @Override // com.google.ar.sceneform.Node.OnAccurateTapListner
                                    public final void onAccurateTap(HitTestResult hitTestResult, MotionEvent motionEvent) {
                                        yd ydVar2 = yd.this;
                                        ve veVar3 = veVar2;
                                        MediaPlayer mediaPlayer2 = mediaPlayer;
                                        Objects.requireNonNull(ydVar2);
                                        if (Objects.equals(veVar3.f5355f, "GOTO_SCENE")) {
                                            ydVar2.C = veVar3.f5356g;
                                            ydVar2.m();
                                        } else if (Objects.equals(veVar3.f5355f, "PLAY_SOUND")) {
                                            Objects.requireNonNull(mediaPlayer2);
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
                        this.f5451b.runOnUiThread(new Runnable() { // from class: c.e.b.x7
                            @Override // java.lang.Runnable
                            public final void run() {
                                final MediaPlayer mediaPlayer;
                                final yd ydVar = yd.this;
                                final ve veVar2 = veVar;
                                Objects.requireNonNull(ydVar);
                                float[] fArr = veVar2.m;
                                float max = Math.max(Math.max(fArr[0], fArr[1]), veVar2.m[2]);
                                Node node = new Node();
                                SimpleTransformableNode simpleTransformableNode = new SimpleTransformableNode(ydVar.f5456g.getTransformationSystem());
                                simpleTransformableNode.setParent(ydVar.f5453d);
                                float[] fArr2 = veVar2.l;
                                simpleTransformableNode.setLocalPosition(new Vector3(fArr2[0], fArr2[1], fArr2[2]));
                                simpleTransformableNode.setLocalRotation(Quaternion.eulerAngles(new Vector3(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD)));
                                node.setParent(simpleTransformableNode);
                                float[] fArr3 = veVar2.q;
                                node.setLocalRotation(Quaternion.multiply(Quaternion.eulerAngles(new Vector3(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 180.0f)), new Quaternion(fArr3[0], fArr3[1], fArr3[2], fArr3[3])));
                                final Node node2 = new Node();
                                node2.setParent(node);
                                node2.setLocalRotation(Quaternion.eulerAngles(new Vector3(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 180.0f, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD)));
                                ydVar.f5456g.getArSceneView().getScene().addOnPeekTouchListener(new Scene.OnPeekTouchListener() { // from class: c.e.b.ca
                                    @Override // com.google.ar.sceneform.Scene.OnPeekTouchListener
                                    public final void onPeekTouch(HitTestResult hitTestResult, MotionEvent motionEvent) {
                                        yd.this.f5456g.getTransformationSystem().onTouch(hitTestResult, motionEvent);
                                    }
                                });
                                if (veVar2.f5355f.equals("PLAY_SOUND")) {
                                    mediaPlayer = ydVar.l();
                                    mediaPlayer.setLooping(veVar2.i);
                                    ydVar.f(veVar2, mediaPlayer);
                                    if (veVar2.f5357h) {
                                        ydVar.I.put(mediaPlayer, Boolean.TRUE);
                                    }
                                } else {
                                    mediaPlayer = null;
                                }
                                if (veVar2.G.equals("LOCK_INTERACTIONS")) {
                                    simpleTransformableNode.setName("LOCK_INTERACTIONS");
                                    float[] fArr4 = veVar2.n;
                                    simpleTransformableNode.setLocalScale(new Vector3(fArr4[0] * 0.08f, fArr4[1] * 0.08f, fArr4[2] * 0.08f).scaled(veVar2.p));
                                } else {
                                    float[] fArr5 = veVar2.n;
                                    node.setLocalScale(new Vector3(fArr5[0], fArr5[1], fArr5[2]).scaled(veVar2.p));
                                }
                                if (max == 1.0f) {
                                    simpleTransformableNode.getScaleController().setMinScale(0.02f);
                                    simpleTransformableNode.getScaleController().setMaxScale(0.14f);
                                } else {
                                    simpleTransformableNode.getScaleController().setMinScale(0.02f * max * 0.08f);
                                    simpleTransformableNode.getScaleController().setMaxScale(max * 0.14f * 0.08f);
                                }
                                String[] split = veVar2.f5354e.split("/");
                                String str = split[split.length - 1];
                                if (str.toLowerCase().endsWith("glb")) {
                                    str = str.replaceAll(".glb", "");
                                }
                                String str2 = ydVar.f5450a.getCacheDir().getPath() + "/assets/models/" + str + ".glb";
                                if (new File(str2).exists()) {
                                    ydVar.t(str2, node2, veVar2, node);
                                } else {
                                    String str3 = veVar2.f5354e;
                                    xd xdVar = new xd(ydVar, node2, veVar2, node);
                                    String[] split2 = str3.split("/");
                                    String str4 = split2[split2.length - 1];
                                    if (str4.toLowerCase().endsWith("glb")) {
                                        str4 = str4.replaceAll(".glb", "");
                                    }
                                    new c.e.b.p000if.k(str4, ydVar.f5450a, xdVar).execute(str3);
                                }
                                node.setOnAccurateTapListner(new Node.OnAccurateTapListner() { // from class: c.e.b.pa
                                    @Override // com.google.ar.sceneform.Node.OnAccurateTapListner
                                    public final void onAccurateTap(HitTestResult hitTestResult, MotionEvent motionEvent) {
                                        yd.this.a(veVar2, mediaPlayer, node2);
                                    }
                                });
                            }
                        });
                        this.N.put(Integer.valueOf(veVar.H), veVar);
                        break;
                    case 7:
                        this.f5451b.runOnUiThread(new Runnable() { // from class: c.e.b.n8
                            @Override // java.lang.Runnable
                            public final void run() {
                                final yd ydVar = yd.this;
                                final ve veVar2 = veVar;
                                Objects.requireNonNull(ydVar);
                                float f2 = veVar2.K.equalsIgnoreCase("fantasy") ? 0.52f : 0.67f;
                                f2 = (veVar2.K.equalsIgnoreCase("cursive") || veVar2.K.equalsIgnoreCase("serif")) ? 0.65f : 0.65f;
                                float f3 = 0.7f;
                                if (veVar2.K.equalsIgnoreCase("monospace") || veVar2.K.equalsIgnoreCase("serif")) {
                                    f3 = 0.82f;
                                } else if (veVar2.K.equalsIgnoreCase("tahoma")) {
                                    f3 = 0.75f;
                                }
                                float f4 = veVar2.m[1];
                                final SimpleTransformableNode simpleTransformableNode = new SimpleTransformableNode(ydVar.f5456g.getTransformationSystem());
                                simpleTransformableNode.setParent(ydVar.f5453d);
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
                                ydVar.f5456g.getArSceneView().getScene().addOnPeekTouchListener(new Scene.OnPeekTouchListener() { // from class: c.e.b.u9
                                    @Override // com.google.ar.sceneform.Scene.OnPeekTouchListener
                                    public final void onPeekTouch(HitTestResult hitTestResult, MotionEvent motionEvent) {
                                        yd.this.f5456g.getTransformationSystem().onTouch(hitTestResult, motionEvent);
                                    }
                                });
                                simpleTransformableNode.getScaleController().setMinScale(0.07f);
                                simpleTransformableNode.getScaleController().setMaxScale(0.8f);
                                if (veVar2.f5354e.length() == 0) {
                                    ViewRenderable.builder().setView(ydVar.f5451b, R.layout.plain_button).build().thenAccept(new Consumer() { // from class: c.e.b.t8
                                        @Override // java.util.function.Consumer
                                        public final void accept(Object obj) {
                                            Typeface a2;
                                            yd ydVar2 = yd.this;
                                            ve veVar3 = veVar2;
                                            Node node = Q;
                                            ViewRenderable viewRenderable = (ViewRenderable) obj;
                                            Objects.requireNonNull(ydVar2);
                                            LinearLayout linearLayout = (LinearLayout) viewRenderable.getView().findViewById(R.id.buttonViewContainers);
                                            TextView textView = (TextView) viewRenderable.getView().findViewById(R.id.button_view_text);
                                            textView.setText(veVar3.r);
                                            if (veVar3.v.length() != 0) {
                                                textView.setTextColor(ydVar2.k(veVar3.v));
                                            }
                                            if (veVar3.x.length() != 0) {
                                                linearLayout.setBackgroundColor(android.graphics.Color.parseColor(veVar3.x.substring(0, 7)));
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
                                                        a2 = b.j.c.b.f.a(ydVar2.f5450a, R.font.sans_serif);
                                                        break;
                                                    case 1:
                                                        a2 = b.j.c.b.f.a(ydVar2.f5450a, R.font.monospace);
                                                        break;
                                                    case 2:
                                                        a2 = b.j.c.b.f.a(ydVar2.f5450a, R.font.fantasy);
                                                        break;
                                                    case 3:
                                                        a2 = b.j.c.b.f.a(ydVar2.f5450a, R.font.tahoma);
                                                        break;
                                                    case 4:
                                                        a2 = b.j.c.b.f.a(ydVar2.f5450a, R.font.georgia);
                                                        break;
                                                    case 5:
                                                        a2 = Typeface.create(veVar3.K, 0);
                                                        break;
                                                    case 6:
                                                        a2 = b.j.c.b.f.a(ydVar2.f5450a, R.font.cursive);
                                                        break;
                                                    default:
                                                        a2 = null;
                                                        break;
                                                }
                                                if (a2 != null) {
                                                    textView.setTypeface(a2);
                                                }
                                            }
                                            viewRenderable.setHorizontalAlignment(ViewRenderable.HorizontalAlignment.CENTER);
                                            c.b.a.a.a.J(viewRenderable, ViewRenderable.VerticalAlignment.CENTER, false, false);
                                            node.setRenderable(viewRenderable);
                                            ydVar2.d(node, veVar3, 0);
                                            ydVar2.K.put(node, veVar3);
                                            int i2 = ydVar2.s - 1;
                                            ydVar2.s = i2;
                                            if (i2 == 0) {
                                                ydVar2.j();
                                            }
                                        }
                                    }).exceptionally(new Function() { // from class: c.e.b.p9
                                        @Override // java.util.function.Function
                                        public final Object apply(Object obj) {
                                            yd ydVar2 = yd.this;
                                            Throwable th = (Throwable) obj;
                                            ydVar2.f5453d.removeChild(simpleTransformableNode);
                                            int i2 = ydVar2.s - 1;
                                            ydVar2.s = i2;
                                            if (i2 == 0) {
                                                ydVar2.j();
                                            }
                                            Log.e("LoaderARContentSceneformARCore", "Unable to load  renderable");
                                            return null;
                                        }
                                    });
                                } else {
                                    ViewRenderable.builder().setView(ydVar.f5451b, R.layout.imagebutton).build().thenAccept(new Consumer() { // from class: c.e.b.ra
                                        @Override // java.util.function.Consumer
                                        public final void accept(Object obj) {
                                            yd ydVar2 = yd.this;
                                            ve veVar3 = veVar2;
                                            Node node = Q;
                                            ViewRenderable viewRenderable = (ViewRenderable) obj;
                                            Objects.requireNonNull(ydVar2);
                                            Log.d("LoaderARContentSceneformARCore", "Building custom image button ViewRenderable");
                                            c.c.a.b.e(ydVar2.f5450a).k(veVar3.f5354e).B((ImageView) viewRenderable.getView().findViewById(R.id.button_view));
                                            viewRenderable.setHorizontalAlignment(ViewRenderable.HorizontalAlignment.CENTER);
                                            c.b.a.a.a.J(viewRenderable, ViewRenderable.VerticalAlignment.CENTER, false, false);
                                            node.setRenderable(viewRenderable);
                                            ydVar2.d(node, veVar3, 0);
                                            ydVar2.K.put(node, veVar3);
                                            int i2 = ydVar2.s - 1;
                                            ydVar2.s = i2;
                                            if (i2 == 0) {
                                                ydVar2.j();
                                            }
                                        }
                                    }).exceptionally(new Function() { // from class: c.e.b.q7
                                        @Override // java.util.function.Function
                                        public final Object apply(Object obj) {
                                            Throwable th = (Throwable) obj;
                                            Objects.requireNonNull(yd.this);
                                            Log.e("LoaderARContentSceneformARCore", "Unable to load  renderable");
                                            return null;
                                        }
                                    });
                                }
                                Q.setOnAccurateTapListner(new Node.OnAccurateTapListner() { // from class: c.e.b.s9
                                    @Override // com.google.ar.sceneform.Node.OnAccurateTapListner
                                    public final void onAccurateTap(HitTestResult hitTestResult, MotionEvent motionEvent) {
                                        String str;
                                        ObjectAnimator objectAnimator;
                                        yd ydVar2 = yd.this;
                                        ve veVar3 = veVar2;
                                        Objects.requireNonNull(ydVar2);
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
                                                ydVar2.C = str2;
                                                ydVar2.m();
                                                return;
                                            case 1:
                                            case 5:
                                                ydVar2.n(str2);
                                                return;
                                            case 2:
                                            case 4:
                                                ydVar2.i(str2);
                                                return;
                                            case 3:
                                                if (ydVar2.H.get(Integer.valueOf(veVar3.P)) == null || (objectAnimator = ydVar2.E.get(Integer.valueOf(veVar3.P))) == null) {
                                                    return;
                                                }
                                                if (objectAnimator.isRunning() || objectAnimator.isPaused()) {
                                                    objectAnimator.pause();
                                                    return;
                                                }
                                                return;
                                            case 6:
                                            case '\t':
                                                ydVar2.v(str2);
                                                return;
                                            case '\b':
                                                if (ydVar2.H.get(Integer.valueOf(veVar3.P)) == null) {
                                                    return;
                                                }
                                                ObjectAnimator objectAnimator2 = ydVar2.E.get(Integer.valueOf(veVar3.P));
                                                if (objectAnimator2 != null) {
                                                    objectAnimator2.pause();
                                                    objectAnimator2.removeAllUpdateListeners();
                                                    objectAnimator2.removeAllListeners();
                                                    objectAnimator2.end();
                                                    objectAnimator2.cancel();
                                                }
                                                Node node = ydVar2.H.get(Integer.valueOf(veVar3.P));
                                                Objects.requireNonNull(node);
                                                ydVar2.e(node.getRenderableInstance(), veVar3.R, veVar3.P, veVar3.Q, true, "trigger");
                                                return;
                                            case '\n':
                                                if (ydVar2.H.get(Integer.valueOf(veVar3.P)) == null) {
                                                    return;
                                                }
                                                ObjectAnimator objectAnimator3 = ydVar2.E.get(Integer.valueOf(veVar3.P));
                                                if (objectAnimator3 != null) {
                                                    String[] strArr = ydVar2.F.get(Integer.valueOf(veVar3.P));
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
                                                    Node node2 = ydVar2.H.get(Integer.valueOf(veVar3.P));
                                                    Objects.requireNonNull(node2);
                                                    ydVar2.e(node2.getRenderableInstance(), veVar3.R, veVar3.P, veVar3.Q, true, "trigger");
                                                    return;
                                                }
                                                Node node3 = ydVar2.H.get(Integer.valueOf(veVar3.P));
                                                Objects.requireNonNull(node3);
                                                ydVar2.e(node3.getRenderableInstance(), veVar3.R, veVar3.P, veVar3.Q, true, "trigger");
                                                return;
                                            case 11:
                                                if (ydVar2.H.get(Integer.valueOf(veVar3.P)) != null) {
                                                    ydVar2.o(ydVar2.L.get(Integer.valueOf(veVar3.P)), ydVar2.H.get(Integer.valueOf(veVar3.P)), (String) ydVar2.O.get(Integer.valueOf(veVar3.P)).get(Integer.valueOf(veVar3.f5353d)), veVar3.P);
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
                        this.f5451b.runOnUiThread(new Runnable() { // from class: c.e.b.a8
                            @Override // java.lang.Runnable
                            public final void run() {
                                final yd ydVar = yd.this;
                                final ve veVar2 = veVar;
                                Objects.requireNonNull(ydVar);
                                float f2 = veVar2.m[1];
                                SimpleTransformableNode simpleTransformableNode = new SimpleTransformableNode(ydVar.f5456g.getTransformationSystem());
                                simpleTransformableNode.setParent(ydVar.f5453d);
                                float[] fArr = veVar2.l;
                                final Node Q = c.b.a.a.a.Q(simpleTransformableNode, new Vector3(fArr[0], fArr[1], fArr[2]), simpleTransformableNode);
                                float[] fArr2 = veVar2.q;
                                Q.setLocalRotation(new Quaternion(-fArr2[0], -fArr2[1], fArr2[2], fArr2[3]));
                                float[] fArr3 = veVar2.n;
                                if (fArr3[0] != StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                                    Q.setLocalScale(new Vector3(fArr3[0], fArr3[1], fArr3[2]).scaled(veVar2.p));
                                }
                                ydVar.f5456g.getArSceneView().getScene().addOnPeekTouchListener(new Scene.OnPeekTouchListener() { // from class: c.e.b.y8
                                    @Override // com.google.ar.sceneform.Scene.OnPeekTouchListener
                                    public final void onPeekTouch(HitTestResult hitTestResult, MotionEvent motionEvent) {
                                        yd.this.f5456g.getTransformationSystem().onTouch(hitTestResult, motionEvent);
                                    }
                                });
                                c.b.a.a.a.c(f2, 0.02f, simpleTransformableNode.getScaleController(), simpleTransformableNode).setMaxScale(f2 * 0.16f);
                                final MediaPlayer l = ydVar.l();
                                ydVar.f(veVar2, l);
                                l.setOnCompletionListener(new MediaPlayer.OnCompletionListener() { // from class: c.e.b.na
                                    @Override // android.media.MediaPlayer.OnCompletionListener
                                    public final void onCompletion(MediaPlayer mediaPlayer) {
                                        yd ydVar2 = yd.this;
                                        ve veVar3 = veVar2;
                                        Objects.requireNonNull(ydVar2);
                                        if (Objects.equals(veVar3.f5355f, "GOTO_SCENE_AFTER_CONTENT_END")) {
                                            ydVar2.C = veVar3.f5356g;
                                            ydVar2.m();
                                        }
                                    }
                                });
                                l.setAudioStreamType(3);
                                try {
                                    l.setScreenOnWhilePlaying(true);
                                    l.setDataSource(veVar2.f5354e);
                                    if (veVar2.o) {
                                        l.setLooping(true);
                                    }
                                    l.prepareAsync();
                                    l.setOnPreparedListener(new MediaPlayer.OnPreparedListener() { // from class: c.e.b.f9
                                        @Override // android.media.MediaPlayer.OnPreparedListener
                                        public final void onPrepared(MediaPlayer mediaPlayer) {
                                            final yd ydVar2 = yd.this;
                                            final ve veVar3 = veVar2;
                                            final Node node = Q;
                                            final MediaPlayer mediaPlayer2 = l;
                                            Objects.requireNonNull(ydVar2);
                                            ViewRenderable.builder().setView(ydVar2.f5451b, R.layout.img_view).build().thenAccept(new Consumer() { // from class: c.e.b.n9
                                                @Override // java.util.function.Consumer
                                                public final void accept(Object obj) {
                                                    yd ydVar3 = yd.this;
                                                    ve veVar4 = veVar3;
                                                    Node node2 = node;
                                                    MediaPlayer mediaPlayer3 = mediaPlayer2;
                                                    ViewRenderable viewRenderable = (ViewRenderable) obj;
                                                    Objects.requireNonNull(ydVar3);
                                                    ImageView imageView = (ImageView) viewRenderable.getView().findViewById(R.id.img_loader_view);
                                                    ydVar3.P = imageView;
                                                    imageView.setImageResource(R.drawable.audio);
                                                    ydVar3.P.setColorFilter(ydVar3.k(veVar4.N), PorterDuff.Mode.MULTIPLY);
                                                    if (veVar4.M.length() != 0) {
                                                        ydVar3.P.setBackgroundColor(android.graphics.Color.parseColor(veVar4.M.substring(0, 7)));
                                                        if (veVar4.M.length() > 7) {
                                                            ydVar3.P.setAlpha(Integer.valueOf(veVar4.M.substring(7, 9), 16).intValue() / 255.0f);
                                                        }
                                                    }
                                                    int i2 = (int) ((ydVar3.f5450a.getResources().getDisplayMetrics().density * 36.0f) + 0.5f);
                                                    ydVar3.P.setPadding(i2, i2, i2, i2);
                                                    viewRenderable.setHorizontalAlignment(ViewRenderable.HorizontalAlignment.CENTER);
                                                    viewRenderable.setVerticalAlignment(ViewRenderable.VerticalAlignment.CENTER);
                                                    node2.setRenderable(viewRenderable);
                                                    ydVar3.d(node2, veVar4, 0);
                                                    ydVar3.K.put(node2, veVar4);
                                                    if (veVar4.j) {
                                                        ydVar3.I.put(mediaPlayer3, Boolean.TRUE);
                                                        ydVar3.P.setImageResource(R.drawable.audio);
                                                    } else {
                                                        ydVar3.P.setImageResource(R.drawable.audio_mute);
                                                    }
                                                    int i3 = ydVar3.s - 1;
                                                    ydVar3.s = i3;
                                                    if (i3 == 0) {
                                                        ydVar3.j();
                                                    }
                                                }
                                            }).exceptionally(new Function() { // from class: c.e.b.a9
                                                @Override // java.util.function.Function
                                                public final Object apply(Object obj) {
                                                    yd ydVar3 = yd.this;
                                                    Throwable th = (Throwable) obj;
                                                    int i2 = ydVar3.s - 1;
                                                    ydVar3.s = i2;
                                                    if (i2 == 0) {
                                                        ydVar3.j();
                                                    }
                                                    Log.e("LoaderARContentSceneformARCore", "Unable to load  renderable");
                                                    return null;
                                                }
                                            });
                                        }
                                    });
                                } catch (IOException e2) {
                                    e2.printStackTrace();
                                }
                                Q.setOnAccurateTapListner(new Node.OnAccurateTapListner() { // from class: c.e.b.aa
                                    @Override // com.google.ar.sceneform.Node.OnAccurateTapListner
                                    public final void onAccurateTap(HitTestResult hitTestResult, MotionEvent motionEvent) {
                                        yd ydVar2 = yd.this;
                                        ve veVar3 = veVar2;
                                        MediaPlayer mediaPlayer = l;
                                        Node node = Q;
                                        Objects.requireNonNull(ydVar2);
                                        Log.i("kkkkkkkkkkkkkk", mediaPlayer + "");
                                        ImageView imageView = (ImageView) ((ViewRenderable) node.getRenderable()).getView().findViewById(R.id.img_loader_view);
                                        if (!Objects.equals(veVar3.f5355f, "PLAY_PAUSE_CONTENT") && !Objects.equals(veVar3.f5355f, "GOTO_SCENE_AFTER_CONTENT_END")) {
                                            if (Objects.equals(veVar3.f5355f, "GOTO_SCENE")) {
                                                ydVar2.C = veVar3.f5356g;
                                                ydVar2.m();
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
                        this.f5451b.runOnUiThread(new Runnable() { // from class: c.e.b.e8
                            @Override // java.lang.Runnable
                            public final void run() {
                                final MediaPlayer mediaPlayer;
                                final yd ydVar = yd.this;
                                final ve veVar2 = veVar;
                                Objects.requireNonNull(ydVar);
                                float f2 = veVar2.m[1];
                                final SimpleTransformableNode simpleTransformableNode = new SimpleTransformableNode(ydVar.f5456g.getTransformationSystem());
                                simpleTransformableNode.setParent(ydVar.f5453d);
                                float[] fArr = veVar2.l;
                                final Node Q = c.b.a.a.a.Q(simpleTransformableNode, new Vector3(fArr[0], fArr[1], fArr[2]), simpleTransformableNode);
                                float[] fArr2 = veVar2.q;
                                Q.setLocalRotation(new Quaternion(-fArr2[0], -fArr2[1], fArr2[2], fArr2[3]));
                                float[] fArr3 = veVar2.n;
                                if (fArr3[0] != StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                                    Q.setLocalScale(new Vector3(fArr3[0], fArr3[1], fArr3[2]).scaled(veVar2.p));
                                }
                                ydVar.f5456g.getArSceneView().getScene().addOnPeekTouchListener(new Scene.OnPeekTouchListener() { // from class: c.e.b.la
                                    @Override // com.google.ar.sceneform.Scene.OnPeekTouchListener
                                    public final void onPeekTouch(HitTestResult hitTestResult, MotionEvent motionEvent) {
                                        yd.this.f5456g.getTransformationSystem().onTouch(hitTestResult, motionEvent);
                                    }
                                });
                                c.b.a.a.a.c(f2, 0.036f, simpleTransformableNode.getScaleController(), simpleTransformableNode).setMaxScale(f2 * 0.36f);
                                if (veVar2.f5355f.equals("PLAY_SOUND")) {
                                    mediaPlayer = ydVar.l();
                                    mediaPlayer.setLooping(veVar2.i);
                                    ydVar.f(veVar2, mediaPlayer);
                                    if (veVar2.f5357h) {
                                        ydVar.I.put(mediaPlayer, Boolean.TRUE);
                                    }
                                } else {
                                    mediaPlayer = null;
                                }
                                Q.setOnAccurateTapListner(new Node.OnAccurateTapListner() { // from class: c.e.b.o9
                                    @Override // com.google.ar.sceneform.Node.OnAccurateTapListner
                                    public final void onAccurateTap(HitTestResult hitTestResult, MotionEvent motionEvent) {
                                        yd ydVar2 = yd.this;
                                        ve veVar3 = veVar2;
                                        MediaPlayer mediaPlayer2 = mediaPlayer;
                                        Objects.requireNonNull(ydVar2);
                                        if (Objects.equals(veVar3.f5355f, "GOTO_SCENE")) {
                                            ydVar2.C = veVar3.f5356g;
                                            ydVar2.m();
                                        } else if (Objects.equals(veVar3.f5355f, "PLAY_SOUND")) {
                                            if (mediaPlayer2.isPlaying()) {
                                                mediaPlayer2.pause();
                                            } else {
                                                mediaPlayer2.start();
                                            }
                                        }
                                    }
                                });
                                ViewRenderable.builder().setView(ydVar.f5451b, R.layout.img_view).build().thenAccept(new Consumer() { // from class: c.e.b.r9
                                    @Override // java.util.function.Consumer
                                    public final void accept(Object obj) {
                                        yd ydVar2 = yd.this;
                                        ve veVar3 = veVar2;
                                        Node node = Q;
                                        ViewRenderable viewRenderable = (ViewRenderable) obj;
                                        Objects.requireNonNull(ydVar2);
                                        c.c.a.b.d(ydVar2.f5451b).k(veVar3.f5354e).B((ImageView) viewRenderable.getView().findViewById(R.id.img_loader_view));
                                        viewRenderable.setHorizontalAlignment(ViewRenderable.HorizontalAlignment.CENTER);
                                        viewRenderable.setVerticalAlignment(ViewRenderable.VerticalAlignment.CENTER);
                                        node.setRenderable(viewRenderable);
                                        ydVar2.d(node, veVar3, 0);
                                        ydVar2.K.put(node, veVar3);
                                        int i2 = ydVar2.s - 1;
                                        ydVar2.s = i2;
                                        if (i2 == 0) {
                                            ydVar2.j();
                                        }
                                    }
                                }).exceptionally(new Function() { // from class: c.e.b.ga
                                    @Override // java.util.function.Function
                                    public final Object apply(Object obj) {
                                        yd ydVar2 = yd.this;
                                        Throwable th = (Throwable) obj;
                                        ydVar2.f5453d.removeChild(simpleTransformableNode);
                                        int i2 = ydVar2.s - 1;
                                        ydVar2.s = i2;
                                        if (i2 == 0) {
                                            ydVar2.j();
                                        }
                                        Log.e("LoaderARContentSceneformARCore", "Unable to load  renderable");
                                        return null;
                                    }
                                });
                            }
                        });
                        break;
                    case 10:
                        this.f5451b.runOnUiThread(new Runnable() { // from class: c.e.b.g9
                            @Override // java.lang.Runnable
                            public final void run() {
                                final MediaPlayer mediaPlayer;
                                final yd ydVar = yd.this;
                                final ve veVar2 = veVar;
                                Objects.requireNonNull(ydVar);
                                float f2 = veVar2.K.equals("fantasy") ? 1.2f : 1.05f;
                                float f3 = veVar2.K.equals("monospace") ? 1.0f : 0.8f;
                                float f4 = veVar2.m[1];
                                final SimpleTransformableNode simpleTransformableNode = new SimpleTransformableNode(ydVar.f5456g.getTransformationSystem());
                                simpleTransformableNode.setParent(ydVar.f5453d);
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
                                ydVar.f5456g.getArSceneView().getScene().addOnPeekTouchListener(new Scene.OnPeekTouchListener() { // from class: c.e.b.j8
                                    @Override // com.google.ar.sceneform.Scene.OnPeekTouchListener
                                    public final void onPeekTouch(HitTestResult hitTestResult, MotionEvent motionEvent) {
                                        yd.this.f5456g.getTransformationSystem().onTouch(hitTestResult, motionEvent);
                                    }
                                });
                                if (veVar2.f5355f.equals("PLAY_SOUND")) {
                                    mediaPlayer = ydVar.l();
                                    mediaPlayer.setLooping(veVar2.i);
                                    ydVar.f(veVar2, mediaPlayer);
                                    if (veVar2.f5357h) {
                                        ydVar.I.put(mediaPlayer, Boolean.TRUE);
                                    }
                                } else {
                                    mediaPlayer = null;
                                }
                                simpleTransformableNode.getScaleController().setMinScale(0.07f);
                                simpleTransformableNode.getScaleController().setMaxScale(0.7f);
                                ViewRenderable.builder().setView(ydVar.f5451b, R.layout.text).build().thenAccept(new Consumer() { // from class: c.e.b.y9
                                    @Override // java.util.function.Consumer
                                    public final void accept(Object obj) {
                                        Typeface a2;
                                        yd ydVar2 = yd.this;
                                        ve veVar3 = veVar2;
                                        Node node = Q;
                                        ViewRenderable viewRenderable = (ViewRenderable) obj;
                                        Objects.requireNonNull(ydVar2);
                                        LinearLayout linearLayout = (LinearLayout) viewRenderable.getView().findViewById(R.id.textViewContainers);
                                        TextView textView = (TextView) viewRenderable.getView().findViewById(R.id.text_views);
                                        String[] split = veVar3.L.split(" ");
                                        StringBuilder sb = new StringBuilder();
                                        int i2 = 0;
                                        while (i2 < split.length) {
                                            sb.append(split[i2]);
                                            sb.append(" ");
                                            i2++;
                                            if (i2 % 5 == 0) {
                                                sb.append("\n");
                                            }
                                        }
                                        textView.setText(sb.toString().trim());
                                        if (veVar3.v.length() != 0) {
                                            textView.setTextColor(ydVar2.k(veVar3.v));
                                        }
                                        if (veVar3.J.length() != 0) {
                                            linearLayout.setBackgroundColor(android.graphics.Color.parseColor(veVar3.J.substring(0, 7)));
                                            if (veVar3.J.length() > 7) {
                                                linearLayout.setAlpha(Integer.valueOf(veVar3.J.substring(7, 9), 16).intValue() / 255.0f);
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
                                                    a2 = b.j.c.b.f.a(ydVar2.f5450a, R.font.sans_serif);
                                                    break;
                                                case 1:
                                                    a2 = b.j.c.b.f.a(ydVar2.f5450a, R.font.monospace);
                                                    break;
                                                case 2:
                                                    a2 = b.j.c.b.f.a(ydVar2.f5450a, R.font.fantasy);
                                                    break;
                                                case 3:
                                                    a2 = b.j.c.b.f.a(ydVar2.f5450a, R.font.tahoma);
                                                    break;
                                                case 4:
                                                    a2 = b.j.c.b.f.a(ydVar2.f5450a, R.font.georgia);
                                                    break;
                                                case 5:
                                                    a2 = Typeface.create(veVar3.K, 0);
                                                    break;
                                                case 6:
                                                    a2 = b.j.c.b.f.a(ydVar2.f5450a, R.font.cursive);
                                                    break;
                                                default:
                                                    a2 = null;
                                                    break;
                                            }
                                            if (a2 != null) {
                                                textView.setTypeface(a2);
                                            }
                                        }
                                        viewRenderable.setHorizontalAlignment(ViewRenderable.HorizontalAlignment.CENTER);
                                        c.b.a.a.a.J(viewRenderable, ViewRenderable.VerticalAlignment.CENTER, false, false);
                                        node.setRenderable(viewRenderable);
                                        ydVar2.d(node, veVar3, 0);
                                        ydVar2.K.put(node, veVar3);
                                        int i3 = ydVar2.s - 1;
                                        ydVar2.s = i3;
                                        if (i3 == 0) {
                                            ydVar2.j();
                                        }
                                    }
                                }).exceptionally(new Function() { // from class: c.e.b.z9
                                    @Override // java.util.function.Function
                                    public final Object apply(Object obj) {
                                        yd ydVar2 = yd.this;
                                        Throwable th = (Throwable) obj;
                                        ydVar2.f5453d.removeChild(simpleTransformableNode);
                                        int i2 = ydVar2.s - 1;
                                        ydVar2.s = i2;
                                        if (i2 == 0) {
                                            ydVar2.j();
                                        }
                                        Log.e("LoaderARContentSceneformARCore", "Unable to load  renderable");
                                        return null;
                                    }
                                });
                                Q.setOnAccurateTapListner(new Node.OnAccurateTapListner() { // from class: c.e.b.q8
                                    @Override // com.google.ar.sceneform.Node.OnAccurateTapListner
                                    public final void onAccurateTap(HitTestResult hitTestResult, MotionEvent motionEvent) {
                                        yd ydVar2 = yd.this;
                                        ve veVar3 = veVar2;
                                        MediaPlayer mediaPlayer2 = mediaPlayer;
                                        Objects.requireNonNull(ydVar2);
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
                                                ydVar2.C = veVar3.f5356g;
                                                ydVar2.m();
                                                return;
                                            case 1:
                                                ydVar2.q();
                                                ydVar2.n(veVar3.f5356g);
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
                                                ydVar2.q();
                                                ydVar2.i(veVar3.f5356g);
                                                return;
                                            case 4:
                                                ydVar2.q();
                                                ydVar2.v(veVar3.f5356g);
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
                        this.f5451b.runOnUiThread(new Runnable() { // from class: c.e.b.l9
                            @Override // java.lang.Runnable
                            public final void run() {
                                yd ydVar = yd.this;
                                ve veVar2 = veVar;
                                Objects.requireNonNull(ydVar);
                                if (Objects.equals(veVar2.B, "POINT_LIGHT")) {
                                    Light build = Light.builder(Light.Type.POINT).setColor(new Color(ydVar.k(veVar2.w))).setShadowCastingEnabled(true).setIntensity(veVar2.F * 300.0f).setFalloffRadius(100.0f).build();
                                    Node node = new Node();
                                    float[] fArr = veVar2.l;
                                    node.setLocalPosition(new Vector3(fArr[0], fArr[1], fArr[2]));
                                    float[] fArr2 = veVar2.q;
                                    node.setLocalRotation(Quaternion.multiply(Quaternion.eulerAngles(new Vector3(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 180.0f)), new Quaternion(fArr2[0], fArr2[1], fArr2[2], fArr2[3])));
                                    float[] fArr3 = veVar2.n;
                                    c.b.a.a.a.I(new Vector3(fArr3[0], fArr3[1], fArr3[2]), veVar2.p, node, build);
                                    node.setParent(ydVar.f5453d);
                                } else if (Objects.equals(veVar2.B, "SPOT_LIGHT")) {
                                    Light.Builder builder = Light.builder(Light.Type.SPOTLIGHT);
                                    Context context = ydVar.f5450a;
                                    Object obj = b.j.c.a.f2074a;
                                    Light build2 = builder.setColor(new Color(context.getColor(R.color.arGalleryAppBarColorDark))).setShadowCastingEnabled(true).setIntensity(veVar2.F * 300.0f).setFalloffRadius(100.0f).build();
                                    Node node2 = new Node();
                                    float[] fArr4 = veVar2.l;
                                    node2.setLocalPosition(new Vector3(fArr4[0], fArr4[1], fArr4[2]));
                                    float[] fArr5 = veVar2.q;
                                    node2.setLocalRotation(Quaternion.multiply(Quaternion.eulerAngles(new Vector3(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 180.0f)), new Quaternion(fArr5[0], fArr5[1], fArr5[2], fArr5[3])));
                                    float[] fArr6 = veVar2.n;
                                    c.b.a.a.a.I(new Vector3(fArr6[0], fArr6[1], fArr6[2]), veVar2.p, node2, build2);
                                    node2.setParent(ydVar.f5453d);
                                } else if (Objects.equals(veVar2.B, "DIRECTIONAL_LIGHT")) {
                                    Light build3 = Light.builder(Light.Type.DIRECTIONAL).setColor(new Color(ydVar.k(veVar2.w))).setShadowCastingEnabled(true).setIntensity(veVar2.F * 300.0f).build();
                                    Node node3 = new Node();
                                    float[] fArr7 = veVar2.l;
                                    node3.setLocalPosition(new Vector3(fArr7[0], fArr7[1], fArr7[2]));
                                    float[] fArr8 = veVar2.q;
                                    node3.setLocalRotation(Quaternion.multiply(Quaternion.eulerAngles(new Vector3(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 180.0f)), new Quaternion(fArr8[0], fArr8[1], fArr8[2], fArr8[3])));
                                    float[] fArr9 = veVar2.n;
                                    c.b.a.a.a.I(new Vector3(fArr9[0], fArr9[1], fArr9[2]), veVar2.p, node3, build3);
                                    node3.setParent(ydVar.f5453d);
                                }
                                int i2 = ydVar.s - 1;
                                ydVar.s = i2;
                                if (i2 == 0) {
                                    ydVar.j();
                                }
                            }
                        });
                        break;
                    default:
                        Log.d("LoaderARContentSceneformARCore", i + " is not supported");
                        break;
                }
            } catch (NumberFormatException unused) {
                int i2 = this.s - 1;
                this.s = i2;
                if (i2 == 0) {
                    j();
                }
            }
        } catch (JSONException e2) {
            e2.printStackTrace();
        }
    }

    public void h() {
        Node node = new Node();
        this.f5454e = node;
        node.setParent(this.f5453d);
        this.f5454e.setLocalPosition(new Vector3(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 0.02f));
        ViewRenderable.builder().setView(this.f5451b, R.layout.image_target_loader).build().thenAccept(new Consumer() { // from class: c.e.b.q9
            @Override // java.util.function.Consumer
            public final void accept(Object obj) {
                yd ydVar = yd.this;
                ViewRenderable viewRenderable = (ViewRenderable) obj;
                Objects.requireNonNull(ydVar);
                ((ProgressBar) viewRenderable.getView().findViewById(R.id.imageTargetLoaderProgressbar)).setIndeterminate(true);
                ((TextView) viewRenderable.getView().findViewById(R.id.imageTargetLoaderProgressbarText)).setText("");
                viewRenderable.setRenderPriority(7);
                viewRenderable.setHorizontalAlignment(ViewRenderable.HorizontalAlignment.CENTER);
                viewRenderable.setVerticalAlignment(ViewRenderable.VerticalAlignment.CENTER);
                ydVar.f5454e.setRenderable(viewRenderable);
                c.b.a.a.a.C(0.12f, 0.12f, 0.12f, ydVar.f5454e);
            }
        }).exceptionally(new Function() { // from class: c.e.b.i9
            @Override // java.util.function.Function
            public final Object apply(Object obj) {
                Throwable th = (Throwable) obj;
                Objects.requireNonNull(yd.this);
                Log.e("LoaderARContentSceneformARCore", "Unable to load  renderable");
                return null;
            }
        });
    }

    public final void i(String str) {
        this.f5450a.startActivity(new Intent("android.intent.action.DIAL", Uri.parse("tel:" + str)));
    }

    public final void j() {
        Node node = this.f5454e;
        if (node != null) {
            node.setParent(null);
            this.f5454e = null;
        }
        this.I.forEach(j9.f4932a);
    }

    public final int k(String str) {
        if (str.length() > 8) {
            str = c.b.a.a.a.r("#", str.substring(str.length() - 2), str.substring(1, str.length() - 2));
        }
        return android.graphics.Color.parseColor(str);
    }

    public final MediaPlayer l() {
        if (this.j == null) {
            MediaPlayer mediaPlayer = new MediaPlayer();
            this.j = mediaPlayer;
            return mediaPlayer;
        } else if (this.k == null) {
            MediaPlayer mediaPlayer2 = new MediaPlayer();
            this.k = mediaPlayer2;
            return mediaPlayer2;
        } else if (this.l == null) {
            MediaPlayer mediaPlayer3 = new MediaPlayer();
            this.l = mediaPlayer3;
            return mediaPlayer3;
        } else if (this.m == null) {
            MediaPlayer mediaPlayer4 = new MediaPlayer();
            this.m = mediaPlayer4;
            return mediaPlayer4;
        } else {
            MediaPlayer mediaPlayer5 = new MediaPlayer();
            this.n = mediaPlayer5;
            return mediaPlayer5;
        }
    }

    public final void m() {
        w();
        this.K.forEach(new BiConsumer() { // from class: c.e.b.p7
            @Override // java.util.function.BiConsumer
            public final void accept(Object obj, Object obj2) {
                yd.this.d((Node) obj, (ve) obj2, 1);
            }
        });
    }

    public final void n(String str) {
        if (!str.startsWith("http://") && !str.startsWith("https://")) {
            str = c.b.a.a.a.q("http://", str);
        }
        this.f5450a.startActivity(new Intent("android.intent.action.VIEW", Uri.parse(str)));
    }

    public void o(ModelRenderable modelRenderable, final Node node, final String str, final int i) {
        if (Objects.equals(this.M.get(Integer.valueOf(i)), str)) {
            return;
        }
        if (!this.L.containsKey(Integer.valueOf(i))) {
            this.L.put(Integer.valueOf(i), modelRenderable);
        }
        if (!Objects.equals(str, "default") && str != null) {
            Texture.builder().setSampler(Texture.Sampler.builder().setMinFilter(Texture.Sampler.MinFilter.LINEAR_MIPMAP_LINEAR).setMagFilter(Texture.Sampler.MagFilter.LINEAR).setWrapMode(Texture.Sampler.WrapMode.REPEAT).build()).setSource(this.f5450a, Uri.parse(this.f5457h.a(str.substring(str.lastIndexOf(47) + 1)))).setUsage(Texture.Usage.DATA).build().thenAccept(new Consumer() { // from class: c.e.b.da
                @Override // java.util.function.Consumer
                public final void accept(Object obj) {
                    yd ydVar = yd.this;
                    Node node2 = node;
                    int i2 = i;
                    String str2 = str;
                    Texture texture = (Texture) obj;
                    Objects.requireNonNull(ydVar);
                    RenderableInstance renderableInstance = node2.getRenderableInstance();
                    int materialsCount = renderableInstance.getMaterialsCount();
                    for (int i3 = 0; i3 < materialsCount; i3++) {
                        renderableInstance.getMaterial(i3).setTexture("baseColorMap", texture);
                    }
                    ydVar.M.put(Integer.valueOf(i2), str2);
                    if (ydVar.E.get(Integer.valueOf(i2)) != null && !ydVar.F.get(Integer.valueOf(i2))[0].toLowerCase().equals("all")) {
                        node2.getRenderableInstance().animate(ydVar.F.get(Integer.valueOf(i2))[0]).start();
                        return;
                    }
                    Node node3 = ydVar.H.get(Integer.valueOf(i2));
                    Objects.requireNonNull(node3);
                    RenderableInstance renderableInstance2 = node3.getRenderableInstance();
                    ve veVar = ydVar.N.get(Integer.valueOf(i2));
                    Objects.requireNonNull(veVar);
                    int i4 = veVar.H;
                    ve veVar2 = ydVar.N.get(Integer.valueOf(i2));
                    Objects.requireNonNull(veVar2);
                    String str3 = veVar2.z;
                    ve veVar3 = ydVar.N.get(Integer.valueOf(i2));
                    Objects.requireNonNull(veVar3);
                    ydVar.e(renderableInstance2, "REPEAT", i4, str3, veVar3.j, "model");
                }
            }).exceptionally(new Function() { // from class: c.e.b.k8
                @Override // java.util.function.Function
                public final Object apply(Object obj) {
                    Throwable th = (Throwable) obj;
                    Objects.requireNonNull(yd.this);
                    Log.e("LoaderARContentSceneformARCore", "Unable to load texture");
                    return null;
                }
            });
            return;
        }
        node.setRenderable(this.L.get(Integer.valueOf(i)).makeCopy());
        this.M.put(Integer.valueOf(i), "default");
        if (this.E.get(Integer.valueOf(i)) != null && !this.F.get(Integer.valueOf(i))[0].toLowerCase().equals("all")) {
            node.getRenderableInstance().animate(this.F.get(Integer.valueOf(i))[0]).start();
            return;
        }
        Node node2 = this.H.get(Integer.valueOf(i));
        Objects.requireNonNull(node2);
        RenderableInstance renderableInstance = node2.getRenderableInstance();
        ve veVar = this.N.get(Integer.valueOf(i));
        Objects.requireNonNull(veVar);
        int i2 = veVar.H;
        ve veVar2 = this.N.get(Integer.valueOf(i));
        Objects.requireNonNull(veVar2);
        String str2 = veVar2.z;
        ve veVar3 = this.N.get(Integer.valueOf(i));
        Objects.requireNonNull(veVar3);
        e(renderableInstance, "REPEAT", i2, str2, veVar3.j, "model");
    }

    public void p(Node node, final int i, final int[] iArr, final String[] strArr, final ImageView imageView, final Node node2, final Node node3) {
        float f2 = i * 0.35f;
        Node node4 = i > 0 ? node3 : node2;
        this.o.add(node4);
        node4.setParent(node);
        node4.setLocalPosition(new Vector3(f2, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 0.01f));
        node4.setOnAccurateTapListner(new Node.OnAccurateTapListner() { // from class: c.e.b.d9
            @Override // com.google.ar.sceneform.Node.OnAccurateTapListner
            public final void onAccurateTap(HitTestResult hitTestResult, MotionEvent motionEvent) {
                yd ydVar = yd.this;
                int[] iArr2 = iArr;
                String[] strArr2 = strArr;
                Node node5 = node2;
                Node node6 = node3;
                ImageView imageView2 = imageView;
                Objects.requireNonNull(ydVar);
                iArr2[0] = iArr2[0] + 1;
                if (iArr2[0] < 0) {
                    iArr2[0] = strArr2.length - 1;
                }
                if (iArr2[0] > strArr2.length - 1) {
                    iArr2[0] = 0;
                }
                c.c.a.b.d(ydVar.f5451b).k(strArr2[iArr2[0]]).C(new wd(ydVar, node5, node6)).B(imageView2);
            }
        });
        final Node node5 = node4;
        ViewRenderable.builder().setView(this.f5451b, R.layout.img_view).build().thenAccept(new Consumer() { // from class: c.e.b.d8
            @Override // java.util.function.Consumer
            public final void accept(Object obj) {
                yd ydVar = yd.this;
                Node node6 = node5;
                int i2 = i;
                Node node7 = node2;
                Node node8 = node3;
                ViewRenderable viewRenderable = (ViewRenderable) obj;
                Objects.requireNonNull(ydVar);
                ((ImageView) viewRenderable.getView().findViewById(R.id.img_loader_view)).setImageResource(R.drawable.slide_button);
                viewRenderable.setRenderPriority(5);
                viewRenderable.setHorizontalAlignment(ViewRenderable.HorizontalAlignment.CENTER);
                viewRenderable.setVerticalAlignment(ViewRenderable.VerticalAlignment.CENTER);
                node6.setRenderable(viewRenderable);
                node6.setLocalScale(new Vector3(i2 * 0.2f, 0.2f, 0.2f));
                node7.setLocalPosition(new Vector3(-ydVar.i, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 0.01f));
                node8.setLocalPosition(new Vector3(ydVar.i, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 0.01f));
            }
        }).exceptionally(new Function() { // from class: c.e.b.sa
            @Override // java.util.function.Function
            public final Object apply(Object obj) {
                Throwable th = (Throwable) obj;
                Objects.requireNonNull(yd.this);
                Log.e("LoaderARContentSceneformARCore", "Unable to load  renderable");
                return null;
            }
        });
    }

    public void q() {
        this.J.clear();
        MediaPlayer mediaPlayer = this.j;
        if (mediaPlayer != null && mediaPlayer.isPlaying()) {
            this.j.pause();
            this.J.put(this.j, Boolean.TRUE);
        }
        MediaPlayer mediaPlayer2 = this.k;
        if (mediaPlayer2 != null && mediaPlayer2.isPlaying()) {
            this.k.pause();
            this.J.put(this.k, Boolean.TRUE);
        }
        MediaPlayer mediaPlayer3 = this.l;
        if (mediaPlayer3 != null && mediaPlayer3.isPlaying()) {
            this.l.pause();
            this.J.put(this.l, Boolean.TRUE);
        }
        MediaPlayer mediaPlayer4 = this.m;
        if (mediaPlayer4 != null && mediaPlayer4.isPlaying()) {
            this.m.pause();
            this.J.put(this.m, Boolean.TRUE);
        }
        MediaPlayer mediaPlayer5 = this.n;
        if (mediaPlayer5 == null || !mediaPlayer5.isPlaying()) {
            return;
        }
        this.n.pause();
        this.J.put(this.n, Boolean.TRUE);
    }

    public void r(ve veVar) {
        q();
        Intent intent = new Intent(this.f5450a, Player360Activity.class);
        intent.putExtra(ImagesContract.URL, veVar.f5354e);
        this.f5450a.startActivity(intent);
    }

    public final void s(Node node) {
        this.f5453d.removeChild(node.getParent());
        if (this.f5453d.getChildren().size() == this.t) {
            if (this.C.length() == 0) {
                this.f5455f.a("Scene not found");
                return;
            }
            this.I.clear();
            e eVar = this.f5455f;
            this.f5455f = eVar;
            String str = !this.u ? "1" : CrashlyticsReportDataCapture.SIGNAL_DEFAULT;
            h();
            this.f5452c = new cc();
            String str2 = this.z + "app/get-ar-new-content/" + this.y + "/" + this.A + "/1/" + str + "/1/" + this.C;
            Log.d("LoaderARContentSceneformARCore", str2);
            this.f5452c.a(str2, new sd(this, eVar));
            this.r = System.currentTimeMillis();
        }
    }

    public final void t(String str, Node node, ve veVar, Node node2) {
        new c.e.b.p000if.m(this.f5450a, new a(veVar, str, node, node2)).execute(veVar);
    }

    public void u(Node node, ve veVar) {
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
            Log.d("LoaderARContentSceneformARCore", x.toString());
            float max = 1.0f / Math.max(Math.max(halfExtent[0], halfExtent[1]), halfExtent[2]);
            float f2 = -max;
            node.setLocalScale(new Vector3(f2, f2, max));
            Log.d("LoaderARContentSceneformARCore", "load3Dmodel bounds " + halfExtent[0] + ", " + halfExtent[1] + ", " + halfExtent[2] + "  Scale = " + max);
            if (veVar == null) {
                return;
            }
            float f3 = center[0] * max;
            float f4 = center[1] * max;
            float f5 = center[2] * max;
            float[] fArr = veVar.l;
            if (fArr[2] == 0.008f || fArr[2] == StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                f4 = (center[1] * max) - (halfExtent[1] * max);
            }
            StringBuilder x2 = c.b.a.a.a.x("load3Dmodel yCorrection ");
            x2.append(node.getLocalPosition());
            x2.append(" correction = ");
            x2.append(f3);
            x2.append(", ");
            x2.append(f4);
            x2.append(", ");
            x2.append(f5);
            Log.d("LoaderARContentSceneformARCore", x2.toString());
            node.setLocalPosition(new Vector3(node.getLocalPosition().x - f3, node.getLocalPosition().y + f4, node.getLocalPosition().z + f5));
        }
    }

    public final void v(String str) {
        Intent intent = new Intent("android.intent.action.SEND");
        intent.setType("text/plain");
        intent.putExtra("android.intent.extra.EMAIL", new String[]{str});
        try {
            this.f5450a.startActivity(intent);
        } catch (Exception e2) {
            Log.e("LoaderARContentSceneformARCore", e2.toString());
            this.f5450a.startActivity(Intent.createChooser(intent, "Send Email"));
        }
    }

    public void w() {
        Log.i("kkkkkkkkkkkkkk", "stppped");
        MediaPlayer mediaPlayer = this.j;
        if (mediaPlayer != null) {
            mediaPlayer.stop();
            this.j.release();
            this.j = null;
        }
        MediaPlayer mediaPlayer2 = this.k;
        if (mediaPlayer2 != null) {
            mediaPlayer2.stop();
            this.k.release();
            this.k = null;
        }
        MediaPlayer mediaPlayer3 = this.l;
        if (mediaPlayer3 != null) {
            mediaPlayer3.stop();
            this.l.release();
            this.l = null;
        }
        MediaPlayer mediaPlayer4 = this.m;
        if (mediaPlayer4 != null) {
            mediaPlayer4.stop();
            this.m.release();
            this.m = null;
        }
        MediaPlayer mediaPlayer5 = this.n;
        if (mediaPlayer5 != null) {
            mediaPlayer5.stop();
            this.n.release();
            this.n = null;
        }
    }

    public final void x(boolean z) {
        Iterator<Node> it = this.o.iterator();
        while (it.hasNext()) {
            it.next().setEnabled(z);
        }
        this.v = z;
    }
}