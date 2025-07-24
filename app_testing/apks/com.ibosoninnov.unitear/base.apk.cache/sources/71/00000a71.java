package c.e.b;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.graphics.Typeface;
import android.media.MediaPlayer;
import android.net.Uri;
import android.os.Handler;
import android.util.Log;
import android.view.MotionEvent;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;
import androidx.cardview.widget.CardView;
import com.google.android.gms.common.internal.ImagesContract;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.google.ar.sceneform.HitTestResult;
import com.google.ar.sceneform.Node;
import com.google.ar.sceneform.Scene;
import com.google.ar.sceneform.math.Quaternion;
import com.google.ar.sceneform.math.Vector3;
import com.google.ar.sceneform.rendering.Color;
import com.google.ar.sceneform.rendering.ExternalTexture;
import com.google.ar.sceneform.rendering.Light;
import com.google.ar.sceneform.rendering.ViewRenderable;
import com.google.ar.sceneform.ux.SimpleFragment;
import com.google.ar.sceneform.ux.SimpleTransformableNode;
import com.ibosoninnov.unitear.R;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.Objects;
import java.util.function.Consumer;
import java.util.function.Function;
import org.json.JSONException;
import org.json.JSONObject;

/* compiled from: LoaderARContentGroundPlaneSceneform.java */
/* loaded from: classes2.dex */
public class jc {

    /* renamed from: a  reason: collision with root package name */
    public static MediaPlayer f4941a;

    /* renamed from: b  reason: collision with root package name */
    public static MediaPlayer f4942b;

    /* renamed from: c  reason: collision with root package name */
    public static MediaPlayer f4943c;

    /* renamed from: d  reason: collision with root package name */
    public String f4944d;

    /* renamed from: e  reason: collision with root package name */
    public String f4945e;

    /* renamed from: f  reason: collision with root package name */
    public ec f4946f;

    /* renamed from: g  reason: collision with root package name */
    public Context f4947g;

    /* renamed from: h  reason: collision with root package name */
    public Activity f4948h;
    public Node i;
    public Node j;
    public Node k;
    public SimpleFragment l;
    public boolean n = false;
    public boolean o = false;
    public ProgressBar r = null;
    public TextView s = null;
    public ArrayList<Node> m = new ArrayList<>();
    public Handler p = new Handler();
    public Runnable q = new b();

    /* compiled from: LoaderARContentGroundPlaneSceneform.java */
    /* loaded from: classes2.dex */
    public class a implements Scene.OnPeekTouchListener {
        public a() {
        }

        @Override // com.google.ar.sceneform.Scene.OnPeekTouchListener
        public void onPeekTouch(HitTestResult hitTestResult, MotionEvent motionEvent) {
            jc.this.l.getTransformationSystem().onTouch(hitTestResult, motionEvent);
        }
    }

    /* compiled from: LoaderARContentGroundPlaneSceneform.java */
    /* loaded from: classes2.dex */
    public class b implements Runnable {
        public b() {
        }

        @Override // java.lang.Runnable
        public void run() {
            jc.a(jc.this, false);
        }
    }

    /* compiled from: LoaderARContentGroundPlaneSceneform.java */
    /* loaded from: classes2.dex */
    public interface c {
    }

    public jc(String str, String str2, Node node, SimpleFragment simpleFragment, Context context, Activity activity) {
        this.f4944d = str;
        this.f4945e = str2;
        this.l = simpleFragment;
        this.i = node;
        this.f4947g = context;
        this.f4948h = activity;
        Node node2 = new Node();
        this.k = node2;
        node2.setParent(node);
        this.k.setLocalRotation(Quaternion.eulerAngles(new Vector3(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 180.0f)));
    }

    public static void a(jc jcVar, boolean z) {
        Iterator<Node> it = jcVar.m.iterator();
        while (it.hasNext()) {
            it.next().setEnabled(z);
        }
        jcVar.n = z;
    }

    public static void b(jc jcVar, JSONObject jSONObject, int i) {
        Objects.requireNonNull(jcVar);
        try {
            String string = jSONObject.getString("ar_content_id");
            int i2 = jSONObject.getInt("typeId");
            if (i2 == 1) {
                jcVar.l(jSONObject, i);
            } else if (i2 == 6) {
                jcVar.k(jSONObject, i);
            } else if (i2 != 7) {
                Log.d("LoaderARContentGroundPlaneSceneform", i2 + " is not supported");
            } else {
                jcVar.f(jSONObject);
            }
            Log.d("LoaderARContentGroundPlaneSceneform", "AR Object " + string + " Type=" + i2);
        } catch (JSONException e2) {
            e2.printStackTrace();
        }
    }

    public static void c(jc jcVar, int i, String str) {
        Objects.requireNonNull(jcVar);
        Log.d("LoaderARContentGroundPlaneSceneform", "buttonAction buttonTypeId = " + i + " data = " + str);
        if (i == 1) {
            Intent intent = new Intent("android.intent.action.DIAL");
            intent.setData(Uri.parse("tel:" + str));
            jcVar.f4947g.startActivity(intent);
        } else if (i != 4) {
            if (!str.startsWith("http://") && !str.startsWith("https://")) {
                str = c.b.a.a.a.q("http://", str);
            }
            jcVar.f4947g.startActivity(new Intent("android.intent.action.VIEW", Uri.parse(str)));
        } else {
            Intent intent2 = new Intent("android.intent.action.SEND");
            intent2.setType("text/plain");
            intent2.putExtra("android.intent.extra.EMAIL", new String[]{str});
            intent2.putExtra("android.intent.extra.TEXT", "");
            try {
                jcVar.f4947g.startActivity(intent2);
            } catch (Exception e2) {
                Log.e("LoaderARContentGroundPlaneSceneform", e2.toString());
                jcVar.f4947g.startActivity(Intent.createChooser(intent2, "Send Email"));
            }
        }
    }

    public static void d(jc jcVar, int i) {
        jcVar.p.postDelayed(jcVar.q, i);
        if (jcVar.o) {
            return;
        }
        jcVar.o = true;
        jcVar.l.getArSceneView().getScene().addOnPeekTouchListener(new mc(jcVar));
    }

    public final void e(float f2) {
        Light build = Light.builder(Light.Type.DIRECTIONAL).setColor(new Color(1.0f, 1.0f, 1.0f)).setShadowCastingEnabled(true).setIntensity(f2 * 200.0f).build();
        this.l.getArSceneView().getScene().getSunlight().setEnabled(false);
        Node node = new Node();
        node.setWorldPosition(new Vector3(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 10.0f, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD));
        node.setLight(build);
        node.setParent(this.i);
        Log.d("LoaderARContentGroundPlaneSceneform", "Light " + build.getIntensity() + " " + this.l.getArSceneView().getScene().getSunlight().getLight().getIntensity());
    }

    public final void f(JSONObject jSONObject) {
        final JSONObject jSONObject2 = jSONObject.getJSONObject("properties");
        JSONObject jSONObject3 = jSONObject2.getJSONObject("scale");
        JSONObject jSONObject4 = jSONObject2.getJSONObject("position");
        jSONObject2.getBoolean("lock3dInteraction");
        final float[] fArr = {((float) jSONObject4.getDouble("x")) * (-0.05f), (((float) jSONObject4.getDouble("z")) * (-0.05f)) - StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD};
        final float f2 = ((float) jSONObject3.getDouble("y")) * ((float) jSONObject2.getDouble("scaleMagnitude"));
        int i = jSONObject2.getInt("buttonTypeId");
        final int i2 = i == 2 ? 1 : i == 3 ? 4 : 3;
        this.f4948h.runOnUiThread(new Runnable() { // from class: c.e.b.a3
            @Override // java.lang.Runnable
            public final void run() {
                jc jcVar = jc.this;
                JSONObject jSONObject5 = jSONObject2;
                float[] fArr2 = fArr;
                float f3 = f2;
                int i3 = i2;
                Objects.requireNonNull(jcVar);
                try {
                    jcVar.g(jSONObject5.getString("buttonData"), !jSONObject5.getBoolean("lock3dInteraction"), fArr2, f3, jSONObject5.getString("buttonText"), jSONObject5.getString("buttonColor"), jSONObject5.getString("buttonTextColor"), i3, "", jSONObject5.getString("fontFamily"));
                } catch (JSONException e2) {
                    e2.printStackTrace();
                }
            }
        });
    }

    public void g(final String str, boolean z, float[] fArr, final float f2, final String str2, final String str3, final String str4, final int i, final String str5, final String str6) {
        Node node = this.j;
        if (node != null) {
            node.setParent(null);
            this.j = null;
        }
        final SimpleTransformableNode simpleTransformableNode = new SimpleTransformableNode(this.l.getTransformationSystem());
        simpleTransformableNode.setParent(this.k);
        simpleTransformableNode.setLocalPosition(new Vector3(fArr[0], fArr[1], fArr[2]));
        simpleTransformableNode.setLocalRotation(Quaternion.eulerAngles(new Vector3(-90.0f, 180.0f, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD)));
        this.l.getArSceneView().getScene().addOnPeekTouchListener(new a());
        if (z) {
            simpleTransformableNode.getRotationController().setEnabled(true);
            simpleTransformableNode.getScaleController().setEnabled(true);
            simpleTransformableNode.getTranslationController().setEnabled(true);
        } else {
            simpleTransformableNode.getRotationController().setEnabled(false);
            simpleTransformableNode.getScaleController().setEnabled(false);
            simpleTransformableNode.getTranslationController().setEnabled(false);
        }
        simpleTransformableNode.getScaleController().setMinScale(0.07f);
        simpleTransformableNode.getScaleController().setMaxScale(0.7f);
        if (str5.isEmpty()) {
            ViewRenderable.builder().setView(this.f4948h, R.layout.button).build().thenAccept(new Consumer() { // from class: c.e.b.g3
                @Override // java.util.function.Consumer
                public final void accept(Object obj) {
                    Typeface create;
                    jc jcVar = jc.this;
                    String str7 = str2;
                    String str8 = str4;
                    String str9 = str3;
                    int i2 = i;
                    String str10 = str;
                    String str11 = str6;
                    SimpleTransformableNode simpleTransformableNode2 = simpleTransformableNode;
                    float f3 = f2;
                    ViewRenderable viewRenderable = (ViewRenderable) obj;
                    Objects.requireNonNull(jcVar);
                    Log.d("LoaderARContentGroundPlaneSceneform", "Building custom button ViewRenderable");
                    CardView cardView = (CardView) viewRenderable.getView().findViewById(R.id.textViewContainers);
                    TextView textView = (TextView) viewRenderable.getView().findViewById(R.id.text_views);
                    textView.setText(str7);
                    if (!str8.isEmpty()) {
                        textView.setTextColor(jcVar.i(str8));
                    }
                    if (!str9.isEmpty()) {
                        textView.setBackgroundColor(jcVar.i(str9));
                    }
                    textView.setOnClickListener(new kc(jcVar, i2, str10));
                    if (!str11.isEmpty()) {
                        String lowerCase = str11.toLowerCase();
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
                                create = Typeface.create(str11, 0);
                                break;
                            case 1:
                                create = Typeface.create(str11, 0);
                                break;
                            case 2:
                                create = b.j.c.b.f.a(jcVar.f4947g, R.font.ultra);
                                break;
                            case 3:
                                create = b.j.c.b.f.a(jcVar.f4947g, R.font.tahoma);
                                break;
                            case 4:
                                create = b.j.c.b.f.a(jcVar.f4947g, R.font.georgia);
                                break;
                            case 5:
                                create = Typeface.create(str11, 0);
                                break;
                            case 6:
                                create = Typeface.create(str11, 0);
                                break;
                            default:
                                create = null;
                                break;
                        }
                        if (create != null) {
                            textView.setTypeface(create);
                        }
                    }
                    viewRenderable.setHorizontalAlignment(ViewRenderable.HorizontalAlignment.CENTER);
                    c.b.a.a.a.J(viewRenderable, ViewRenderable.VerticalAlignment.CENTER, false, false);
                    simpleTransformableNode2.setRenderable(viewRenderable);
                    float f4 = f3 * 0.15f;
                    simpleTransformableNode2.setLocalScale(new Vector3(f4, f4, f4));
                }
            }).exceptionally(new Function() { // from class: c.e.b.b3
                @Override // java.util.function.Function
                public final Object apply(Object obj) {
                    Throwable th = (Throwable) obj;
                    Objects.requireNonNull(jc.this);
                    Log.e("LoaderARContentGroundPlaneSceneform", "Unable to load  renderable");
                    return null;
                }
            });
        } else {
            ViewRenderable.builder().setView(this.f4948h, R.layout.imagebutton).build().thenAccept(new Consumer() { // from class: c.e.b.e3
                @Override // java.util.function.Consumer
                public final void accept(Object obj) {
                    jc jcVar = jc.this;
                    String str7 = str5;
                    int i2 = i;
                    String str8 = str;
                    SimpleTransformableNode simpleTransformableNode2 = simpleTransformableNode;
                    float f3 = f2;
                    ViewRenderable viewRenderable = (ViewRenderable) obj;
                    Objects.requireNonNull(jcVar);
                    Log.d("LoaderARContentGroundPlaneSceneform", "Building custom image button ViewRenderable");
                    ImageView imageView = (ImageView) viewRenderable.getView().findViewById(R.id.button_view);
                    c.c.a.b.e(jcVar.f4947g).k(str7).B(imageView);
                    imageView.setOnClickListener(new lc(jcVar, i2, str8));
                    viewRenderable.setHorizontalAlignment(ViewRenderable.HorizontalAlignment.CENTER);
                    c.b.a.a.a.J(viewRenderable, ViewRenderable.VerticalAlignment.CENTER, false, false);
                    simpleTransformableNode2.setRenderable(viewRenderable);
                    float f4 = f3 * 0.12f;
                    simpleTransformableNode2.setLocalScale(new Vector3(f4, f4, f4));
                }
            }).exceptionally(new Function() { // from class: c.e.b.c3
                @Override // java.util.function.Function
                public final Object apply(Object obj) {
                    Throwable th = (Throwable) obj;
                    Objects.requireNonNull(jc.this);
                    Log.e("LoaderARContentGroundPlaneSceneform", "Unable to load  renderable");
                    return null;
                }
            });
        }
    }

    public void h(String str, boolean z, boolean z2, boolean z3, float f2, float[] fArr, float[] fArr2, float[] fArr3) {
        MediaPlayer mediaPlayer;
        SimpleTransformableNode simpleTransformableNode = new SimpleTransformableNode(this.l.getTransformationSystem());
        simpleTransformableNode.setParent(this.k);
        simpleTransformableNode.setLocalPosition(new Vector3(fArr[0], fArr[1], fArr[2]));
        Quaternion eulerAngles = Quaternion.eulerAngles(new Vector3(-fArr2[0], -fArr2[1], fArr2[2]));
        Node node = new Node();
        node.setParent(simpleTransformableNode);
        node.setLocalRotation(Quaternion.multiply(Quaternion.eulerAngles(new Vector3(-90.0f, 180.0f, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD)), eulerAngles));
        this.l.getArSceneView().getScene().addOnPeekTouchListener(new qc(this));
        if (z) {
            simpleTransformableNode.getRotationController().setEnabled(true);
            simpleTransformableNode.getScaleController().setEnabled(true);
            simpleTransformableNode.getTranslationController().setEnabled(true);
        } else {
            simpleTransformableNode.getRotationController().setEnabled(false);
            simpleTransformableNode.getScaleController().setEnabled(false);
            simpleTransformableNode.getTranslationController().setEnabled(false);
        }
        c.b.a.a.a.c(f2, 0.01f, simpleTransformableNode.getScaleController(), simpleTransformableNode).setMaxScale(f2 * 0.08f);
        simpleTransformableNode.addLifecycleListener(new rc(this));
        if (f4941a == null) {
            mediaPlayer = new MediaPlayer();
            f4941a = mediaPlayer;
        } else if (f4942b == null) {
            mediaPlayer = new MediaPlayer();
            f4942b = mediaPlayer;
        } else {
            mediaPlayer = new MediaPlayer();
            f4943c = mediaPlayer;
        }
        ExternalTexture externalTexture = new ExternalTexture();
        mediaPlayer.setSurface(externalTexture.getSurface());
        mediaPlayer.setAudioStreamType(3);
        try {
            mediaPlayer.setScreenOnWhilePlaying(true);
            mediaPlayer.setDataSource(str);
            mediaPlayer.setLooping(true);
            mediaPlayer.prepareAsync();
            mediaPlayer.setOnPreparedListener(new sc(this, z3, fArr3, externalTexture, node, simpleTransformableNode, mediaPlayer, z2, str));
        } catch (IOException e2) {
            e2.printStackTrace();
        }
    }

    public final int i(String str) {
        if (str.length() > 8) {
            str = c.b.a.a.a.r("#", str.substring(str.length() - 2), str.substring(1, str.length() - 2));
        }
        Log.d("LoaderARContentGroundPlaneSceneform", "Color " + str);
        return android.graphics.Color.parseColor(str);
    }

    public void j(String str, boolean z, float f2, float[] fArr, float[] fArr2) {
        Node node = new Node();
        SimpleTransformableNode simpleTransformableNode = new SimpleTransformableNode(this.l.getTransformationSystem());
        simpleTransformableNode.setParent(this.k);
        simpleTransformableNode.setLocalPosition(new Vector3(fArr[0], fArr[1], fArr[2]));
        simpleTransformableNode.setLocalRotation(Quaternion.eulerAngles(new Vector3(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD)));
        node.setParent(simpleTransformableNode);
        node.setLocalRotation(Quaternion.multiply(Quaternion.eulerAngles(new Vector3(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD)), Quaternion.eulerAngles(new Vector3(-fArr2[2], fArr2[0], -fArr2[1]))));
        Node node2 = new Node();
        node2.setParent(node);
        node2.setLocalRotation(Quaternion.eulerAngles(new Vector3(90.0f, 180.0f, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD)));
        this.l.getArSceneView().getScene().addOnPeekTouchListener(new oc(this));
        if (z) {
            simpleTransformableNode.getRotationController().setEnabled(true);
            simpleTransformableNode.getScaleController().setEnabled(true);
            simpleTransformableNode.getTranslationController().setEnabled(true);
        } else {
            simpleTransformableNode.getRotationController().setEnabled(false);
            simpleTransformableNode.getScaleController().setEnabled(false);
            simpleTransformableNode.getTranslationController().setEnabled(false);
            float f3 = f2 * 0.1f;
            simpleTransformableNode.setLocalScale(new Vector3(f3, f3, f3));
        }
        c.b.a.a.a.c(f2, 0.01f, simpleTransformableNode.getScaleController(), simpleTransformableNode).setMaxScale(f2 * 0.1f);
        Node node3 = this.j;
        if (node3 != null) {
            ViewRenderable viewRenderable = (ViewRenderable) node3.getRenderable();
            ProgressBar progressBar = (ProgressBar) viewRenderable.getView().findViewById(R.id.imageTargetLoaderProgressbar);
            this.r = progressBar;
            progressBar.setIndeterminate(false);
            this.r.setProgress(0);
            TextView textView = (TextView) viewRenderable.getView().findViewById(R.id.imageTargetLoaderProgressbarText);
            this.s = textView;
            textView.setText("0 %");
        }
        pc pcVar = new pc(this, node2);
        String[] split = str.split("/");
        String str2 = split[split.length - 1];
        if (str2.toLowerCase().endsWith("glb")) {
            str2 = str2.replaceAll(".glb", "");
        }
        new c.e.b.p000if.k(str2, this.f4947g, pcVar).execute(str);
    }

    public final void k(JSONObject jSONObject, final int i) {
        final JSONObject jSONObject2 = jSONObject.getJSONObject("properties");
        JSONObject jSONObject3 = jSONObject2.getJSONObject("scale");
        JSONObject jSONObject4 = jSONObject2.getJSONObject("position");
        JSONObject jSONObject5 = jSONObject2.getJSONObject("rotationQuaternion");
        JSONObject jSONObject6 = jSONObject2.getJSONObject("rotation");
        final boolean z = jSONObject2.getBoolean("lock3dInteraction");
        float f2 = jSONObject2.has("brightness") ? (float) jSONObject2.getDouble("brightness") : 1.0f;
        final float[] fArr = {((float) jSONObject4.getDouble("x")) * (-0.05f), (((float) jSONObject4.getDouble("z")) * (-0.05f)) - StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD};
        jSONObject5.getDouble("x");
        jSONObject5.getDouble("y");
        jSONObject5.getDouble("z");
        jSONObject5.getDouble("w");
        final float[] fArr2 = {(float) jSONObject6.getDouble("x"), (float) jSONObject6.getDouble("y"), (float) jSONObject6.getDouble("z")};
        final float f3 = (float) jSONObject2.getDouble("scaleMagnitude");
        Math.max(Math.max((float) jSONObject3.getDouble("x"), (float) jSONObject3.getDouble("y")), (float) jSONObject3.getDouble("z"));
        Log.d("LoaderARContentGroundPlaneSceneform", "load3Dmodel scale " + f3);
        final float f4 = f2 * f2;
        this.f4948h.runOnUiThread(new Runnable() { // from class: c.e.b.l3
            @Override // java.lang.Runnable
            public final void run() {
                jc jcVar = jc.this;
                float f5 = f4;
                JSONObject jSONObject7 = jSONObject2;
                boolean z2 = z;
                float f6 = f3;
                float[] fArr3 = fArr;
                float[] fArr4 = fArr2;
                Objects.requireNonNull(jcVar);
                try {
                    jcVar.e(f5);
                    jcVar.j(jSONObject7.getString(ImagesContract.URL), !z2, f6, fArr3, fArr4);
                } catch (JSONException e2) {
                    e2.printStackTrace();
                }
            }
        });
    }

    public final void l(JSONObject jSONObject, final int i) {
        final JSONObject jSONObject2 = jSONObject.getJSONObject("properties");
        JSONObject jSONObject3 = jSONObject2.getJSONObject("scale");
        JSONObject jSONObject4 = jSONObject2.getJSONObject("position");
        JSONObject jSONObject5 = jSONObject2.getJSONObject("rotation");
        final boolean z = jSONObject2.getBoolean("lock3dInteraction");
        final boolean z2 = jSONObject2.has("autoplay") ? jSONObject2.getBoolean("autoplay") : true;
        final float[] fArr = {((float) jSONObject4.getDouble("x")) * (-0.05f), (((float) jSONObject4.getDouble("z")) * (-0.05f)) - StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD};
        final float[] fArr2 = {(float) jSONObject5.getDouble("x"), (float) jSONObject5.getDouble("y"), (float) jSONObject5.getDouble("z")};
        final float f2 = ((float) jSONObject3.getDouble("y")) * 0.5f * ((float) jSONObject2.getDouble("scaleMagnitude"));
        final float[] fArr3 = {StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 1.0f, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD};
        this.f4948h.runOnUiThread(new Runnable() { // from class: c.e.b.j3
            @Override // java.lang.Runnable
            public final void run() {
                jc jcVar = jc.this;
                JSONObject jSONObject6 = jSONObject2;
                boolean z3 = z;
                boolean z4 = z2;
                float f3 = f2;
                float[] fArr4 = fArr;
                float[] fArr5 = fArr2;
                float[] fArr6 = fArr3;
                Objects.requireNonNull(jcVar);
                try {
                    jcVar.h(jSONObject6.getString(ImagesContract.URL), !z3, z4, jSONObject6.getBoolean("alpha"), f3, fArr4, fArr5, fArr6);
                } catch (JSONException e2) {
                    e2.printStackTrace();
                }
            }
        });
    }
}