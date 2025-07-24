package com.ibosoninnov.unitear;

import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.media.MediaPlayer;
import android.os.Bundle;
import android.os.Handler;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.widget.ImageButton;
import android.widget.RelativeLayout;
import b.b.c.h;
import c.e.b.se;
import c.e.b.te;
import c.e.b.ue;
import com.google.android.gms.common.internal.ImagesContract;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.google.ar.sceneform.FrameTime;
import com.google.ar.sceneform.HitTestResult;
import com.google.ar.sceneform.Node;
import com.google.ar.sceneform.Scene;
import com.google.ar.sceneform.SimpleSceneView;
import com.google.ar.sceneform.math.Quaternion;
import com.google.ar.sceneform.math.Vector3;
import com.google.ar.sceneform.rendering.ExternalTexture;
import com.google.ar.sceneform.ux.SimpleFragment;
import f.v;
import f.x;
import f.y;
import java.io.IOException;

/* loaded from: classes2.dex */
public class Player360Activity extends h implements SensorEventListener, Scene.OnUpdateListener, Scene.OnPeekTouchListener {
    public static MediaPlayer r;
    public Handler A;
    public Runnable B;
    public boolean C;
    public Node D;
    public float E;
    public float F;
    public Quaternion G;
    public SensorManager H;
    public float[] I;
    public SimpleFragment t;
    public SimpleSceneView u;
    public RelativeLayout x;
    public RelativeLayout y;
    public ImageButton z;
    public String s = Player360Activity.class.getName();
    public String v = "";
    public boolean w = false;
    public Quaternion J = null;

    /* loaded from: classes2.dex */
    public class a implements View.OnClickListener {
        public a() {
        }

        @Override // android.view.View.OnClickListener
        public void onClick(View view) {
            MediaPlayer mediaPlayer = Player360Activity.r;
            if (mediaPlayer != null) {
                if (mediaPlayer.isPlaying()) {
                    Player360Activity.r.pause();
                    Player360Activity.this.z.setImageResource(R.drawable.play);
                    return;
                }
                Player360Activity.r.start();
                Player360Activity.this.z.setImageResource(R.drawable.pause);
            }
        }
    }

    /* loaded from: classes2.dex */
    public class b implements View.OnClickListener {
        public b() {
        }

        @Override // android.view.View.OnClickListener
        public void onClick(View view) {
            Player360Activity player360Activity = Player360Activity.this;
            MediaPlayer mediaPlayer = Player360Activity.r;
            player360Activity.v();
            Player360Activity.this.finish();
        }
    }

    /* loaded from: classes2.dex */
    public class c implements Runnable {
        public c() {
        }

        @Override // java.lang.Runnable
        public void run() {
            Player360Activity player360Activity = Player360Activity.this;
            player360Activity.y.setVisibility(8);
            player360Activity.C = false;
        }
    }

    @Override // android.hardware.SensorEventListener
    public void onAccuracyChanged(Sensor sensor, int i) {
    }

    @Override // androidx.activity.ComponentActivity, android.app.Activity
    public void onBackPressed() {
        v();
        this.f41f.b();
    }

    @Override // b.b.c.h, b.q.b.d, androidx.activity.ComponentActivity, b.j.b.e, android.app.Activity
    public void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        setContentView(R.layout.activity_player360);
        Bundle extras = getIntent().getExtras();
        if (extras != null && extras.containsKey(ImagesContract.URL)) {
            this.v = extras.getString(ImagesContract.URL);
        }
        this.x = (RelativeLayout) findViewById(R.id.loaderLayout);
        this.y = (RelativeLayout) findViewById(R.id.playerControlsLayout);
        ImageButton imageButton = (ImageButton) findViewById(R.id.playPauseButton);
        this.z = imageButton;
        imageButton.setOnClickListener(new a());
        ((ImageButton) findViewById(R.id.closeButton)).setOnClickListener(new b());
        this.A = new Handler();
        this.B = new c();
        SimpleFragment simpleFragment = (SimpleFragment) m().H(R.id.simple_sceneform_fragment);
        this.t = simpleFragment;
        simpleFragment.getArSceneView().getScene().addOnUpdateListener(this);
        this.t.getArSceneView().getScene().addOnPeekTouchListener(this);
        SimpleSceneView arSceneView = this.t.getArSceneView();
        this.u = arSceneView;
        arSceneView.getScene().getCamera().setVerticalFovDegrees(60.0f);
        if (!this.v.toLowerCase().endsWith("jpeg") && !this.v.toLowerCase().endsWith("jpg") && !this.v.toLowerCase().endsWith("png")) {
            String str = this.v;
            if (!str.isEmpty()) {
                Node node = new Node();
                this.D = node;
                node.setParent(this.u.getScene());
                this.D.setLocalPosition(new Vector3(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD));
                this.D.setEnabled(false);
                this.D.addLifecycleListener(new te(this));
                r = new MediaPlayer();
                ExternalTexture externalTexture = new ExternalTexture();
                r.setSurface(externalTexture.getSurface());
                r.setAudioStreamType(3);
                try {
                    r.setScreenOnWhilePlaying(true);
                    r.setDataSource(str);
                    r.setLooping(true);
                    r.prepareAsync();
                    r.setOnPreparedListener(new ue(this, externalTexture, true));
                } catch (IOException e2) {
                    e2.printStackTrace();
                }
            }
        } else {
            this.w = true;
            String str2 = this.v;
            y.a aVar = new y.a();
            aVar.d(str2);
            ((x) new v().a(aVar.a())).b(new se(this));
        }
        this.H = (SensorManager) getSystemService("sensor");
    }

    @Override // b.b.c.h, b.q.b.d, android.app.Activity
    public void onDestroy() {
        super.onDestroy();
        v();
        Log.d(this.s, "OnDestroy");
    }

    @Override // b.q.b.d, android.app.Activity
    public void onPause() {
        super.onPause();
        this.H.unregisterListener(this);
    }

    @Override // com.google.ar.sceneform.Scene.OnPeekTouchListener
    public void onPeekTouch(HitTestResult hitTestResult, MotionEvent motionEvent) {
        if (motionEvent.getAction() == 0) {
            this.E = motionEvent.getX();
            this.F = motionEvent.getY();
            this.G = this.D.getLocalRotation();
        }
        if (motionEvent.getAction() == 2) {
            float x = motionEvent.getX();
            motionEvent.getX();
            float f2 = this.E - x;
            if (Math.abs(f2) > 5.0f) {
                this.D.setLocalRotation(Quaternion.multiply(this.G, Quaternion.axisAngle(Vector3.up(), f2 * 0.1f)));
            }
        }
        if (this.w) {
            return;
        }
        if (!this.C) {
            this.y.setVisibility(0);
            this.C = true;
        } else if (motionEvent.getAction() == 1) {
            this.A.postDelayed(this.B, 2000L);
        }
    }

    @Override // b.q.b.d, android.app.Activity
    public void onResume() {
        super.onResume();
        SensorManager sensorManager = this.H;
        sensorManager.registerListener(this, sensorManager.getDefaultSensor(11), 1);
    }

    @Override // android.hardware.SensorEventListener
    public void onSensorChanged(SensorEvent sensorEvent) {
        if (sensorEvent.sensor.getType() == 11) {
            float[] fArr = new float[16];
            this.I = fArr;
            SensorManager.getRotationMatrixFromVector(fArr, sensorEvent.values);
            float[] fArr2 = this.I;
            Vector3 vector3 = new Vector3(-fArr2[2], -fArr2[6], -fArr2[10]);
            float[] fArr3 = this.I;
            Vector3 vector32 = new Vector3(fArr3[1], fArr3[5], fArr3[9]);
            float[] fArr4 = this.I;
            new Vector3(fArr4[0], fArr4[4], fArr4[8]);
            Quaternion lookRotation = Quaternion.lookRotation(vector3, vector32);
            if (this.J == null) {
                this.J = lookRotation.inverted();
            }
            this.u.getScene().getCamera().setLocalRotation(Quaternion.slerp(this.u.getScene().getCamera().getLocalRotation(), Quaternion.multiply(this.J, lookRotation), 0.2f));
        }
    }

    @Override // com.google.ar.sceneform.Scene.OnUpdateListener
    public void onUpdate(FrameTime frameTime) {
    }

    public final void v() {
        MediaPlayer mediaPlayer = r;
        if (mediaPlayer != null) {
            mediaPlayer.stop();
            r = null;
        }
        Node node = this.D;
        if (node != null) {
            node.setParent(null);
        }
    }
}