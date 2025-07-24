package com.ibosoninnov.unitear;

import android.net.Uri;
import android.os.Bundle;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.Toast;
import b.b.c.h;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.google.ar.sceneform.Node;
import com.google.ar.sceneform.Scene;
import com.google.ar.sceneform.SimpleSceneView;
import com.google.ar.sceneform.math.Quaternion;
import com.google.ar.sceneform.math.Vector3;
import com.google.ar.sceneform.rendering.ExternalTexture;
import com.google.ar.sceneform.rendering.Material;
import com.google.ar.sceneform.rendering.ModelRenderable;
import com.google.ar.sceneform.rendering.ShapeFactory;
import com.google.ar.sceneform.ux.SimpleFragment;
import com.ibosoninnov.instanttrackinglib.InstantTrackingHelper;
import com.ibosoninnov.unitear.InstantTrackingActivity;
import java.util.List;
import java.util.Objects;
import java.util.function.Consumer;
import java.util.function.Function;

/* loaded from: classes2.dex */
public class InstantTrackingActivity extends h {
    public static final /* synthetic */ int r = 0;
    public ExternalTexture A;
    public float B;
    public float C;
    public InstantTrackingHelper t;
    public SimpleFragment u;
    public SimpleSceneView v;
    public Scene w;
    public Node x;
    public Node y;
    public String s = InstantTrackingActivity.class.getName();
    public Quaternion z = Quaternion.eulerAngles(new Vector3(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 90.0f));

    /* loaded from: classes2.dex */
    public class a implements InstantTrackingHelper.TrackingListener {
        public a() {
        }

        @Override // com.ibosoninnov.instanttrackinglib.InstantTrackingHelper.TrackingListener
        public void onTracking(List<Float> list) {
            if (list != null) {
                InstantTrackingActivity instantTrackingActivity = InstantTrackingActivity.this;
                int i = InstantTrackingActivity.r;
                Objects.requireNonNull(instantTrackingActivity);
                instantTrackingActivity.y.setWorldPosition(new Vector3(list.get(3).floatValue() * 0.2f, (-list.get(7).floatValue()) * 0.2f, list.get(11).floatValue() * 0.2f));
                Vector3 vector3 = new Vector3(list.get(0).floatValue(), -list.get(4).floatValue(), list.get(8).floatValue());
                new Vector3(-list.get(1).floatValue(), list.get(5).floatValue(), list.get(9).floatValue());
                Quaternion lookRotation = Quaternion.lookRotation(vector3, new Vector3(-list.get(2).floatValue(), list.get(6).floatValue(), -list.get(10).floatValue()));
                Node node = instantTrackingActivity.y;
                node.setWorldRotation(Quaternion.slerp(node.getWorldRotation(), Quaternion.multiply(lookRotation, instantTrackingActivity.z), 0.5f));
            }
        }
    }

    /* loaded from: classes2.dex */
    public class b implements View.OnClickListener {
        public b() {
        }

        @Override // android.view.View.OnClickListener
        public void onClick(View view) {
            InstantTrackingActivity.this.t.resetAnchor(0.5f, 0.5f);
        }
    }

    @Override // b.b.c.h, b.q.b.d, androidx.activity.ComponentActivity, b.j.b.e, android.app.Activity
    public void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        setContentView(R.layout.activity_instant_tracking);
        DisplayMetrics displayMetrics = new DisplayMetrics();
        getWindowManager().getDefaultDisplay().getMetrics(displayMetrics);
        this.C = displayMetrics.heightPixels;
        this.B = displayMetrics.widthPixels;
        SimpleFragment simpleFragment = (SimpleFragment) m().H(R.id.simple_sceneform_fragment);
        this.u = simpleFragment;
        SimpleSceneView arSceneView = simpleFragment.getArSceneView();
        this.v = arSceneView;
        this.w = arSceneView.getScene();
        Node node = new Node();
        this.x = node;
        node.setWorldPosition(new Vector3(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, -10.0f));
        this.x.setParent(this.v.getScene().getCamera());
        this.v.getScene().getCamera().setVerticalFovDegrees(66.0f);
        this.A = new ExternalTexture();
        v();
        SimpleSceneView simpleSceneView = this.v;
        Node node2 = new Node();
        this.y = node2;
        node2.setParent(simpleSceneView.getScene());
        this.y.setWorldPosition(new Vector3(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD));
        Vector3 zero = Vector3.zero();
        Vector3 vector3 = new Vector3(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 0.1f);
        Vector3 worldPosition = this.w.getCamera().getWorldPosition();
        Quaternion.lookRotation(Vector3.subtract(new Vector3(zero.x, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, zero.z), new Vector3(worldPosition.x, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, worldPosition.z)), Vector3.up());
        Vector3 add = Vector3.add(zero, vector3);
        final Node node3 = new Node();
        node3.setParent(this.y);
        node3.setWorldPosition(add);
        node3.setLocalScale(new Vector3(0.5f, 0.5f, 0.5f));
        ModelRenderable.builder().setSource(this, Uri.parse("https://storage.googleapis.com/ar-answers-in-search-models/static/Tiger/model.glb")).setIsFilamentGltf(true).build().thenAccept(new Consumer() { // from class: c.e.b.r2
            @Override // java.util.function.Consumer
            public final void accept(Object obj) {
                int i = InstantTrackingActivity.r;
                Node.this.setRenderable((ModelRenderable) obj);
            }
        }).exceptionally(new Function() { // from class: c.e.b.q2
            @Override // java.util.function.Function
            public final Object apply(Object obj) {
                InstantTrackingActivity instantTrackingActivity = InstantTrackingActivity.this;
                Throwable th = (Throwable) obj;
                Objects.requireNonNull(instantTrackingActivity);
                Toast makeText = Toast.makeText(instantTrackingActivity, "Unable to load model", 1);
                makeText.setGravity(17, 0, 0);
                makeText.show();
                return null;
            }
        });
        InstantTrackingHelper instantTrackingHelper = new InstantTrackingHelper(this, this, (ViewGroup) findViewById(R.id.preview_display_layout), this.A.getSurface(), this.A.getSurfaceTexture(), new a());
        this.t = instantTrackingHelper;
        instantTrackingHelper.startTracking();
        ((Button) findViewById(R.id.resetButton)).setOnClickListener(new b());
    }

    @Override // b.b.c.h, b.q.b.d, android.app.Activity
    public void onDestroy() {
        super.onDestroy();
        this.t.stopTracking();
    }

    @Override // b.q.b.d, android.app.Activity
    public void onPause() {
        super.onPause();
        this.t.onPause();
    }

    @Override // b.q.b.d, android.app.Activity
    public void onResume() {
        super.onResume();
        this.t.onResume();
    }

    public final void v() {
        Material.builder().setSource(this, R.raw.augmented_video_material).build().thenAccept(new Consumer() { // from class: c.e.b.t2
            @Override // java.util.function.Consumer
            public final void accept(Object obj) {
                InstantTrackingActivity instantTrackingActivity = InstantTrackingActivity.this;
                instantTrackingActivity.x.setRenderable(ShapeFactory.makeCube(new Vector3(6.4f, (instantTrackingActivity.C / instantTrackingActivity.B) * 6.4f, 0.01f), Vector3.zero(), (Material) obj));
                instantTrackingActivity.x.getRenderable().getMaterial().setExternalTexture("videoTexture", instantTrackingActivity.A);
            }
        }).exceptionally(new Function() { // from class: c.e.b.s2
            @Override // java.util.function.Function
            public final Object apply(Object obj) {
                Throwable th = (Throwable) obj;
                Log.e(InstantTrackingActivity.this.s, "Unable to load camera renderable");
                return null;
            }
        });
    }
}