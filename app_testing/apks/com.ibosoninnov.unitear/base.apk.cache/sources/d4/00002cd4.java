package com.ibosoninnov.unitear;

import android.app.Activity;
import android.media.MediaPlayer;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.widget.MediaController;
import android.widget.ProgressBar;
import android.widget.VideoView;

/* loaded from: classes2.dex */
public class VideoActivity extends Activity {

    /* renamed from: b  reason: collision with root package name */
    public VideoView f5701b;

    /* renamed from: d  reason: collision with root package name */
    public boolean f5703d;

    /* renamed from: f  reason: collision with root package name */
    public ProgressBar f5705f;

    /* renamed from: c  reason: collision with root package name */
    public String f5702c = "";

    /* renamed from: e  reason: collision with root package name */
    public int f5704e = 0;

    /* loaded from: classes2.dex */
    public class a implements MediaPlayer.OnPreparedListener {
        public a() {
        }

        @Override // android.media.MediaPlayer.OnPreparedListener
        public void onPrepared(MediaPlayer mediaPlayer) {
            mediaPlayer.setLooping(VideoActivity.this.f5703d);
            VideoActivity videoActivity = VideoActivity.this;
            videoActivity.f5701b.seekTo(videoActivity.f5704e);
            VideoActivity.this.f5701b.start();
            VideoActivity.this.f5705f.setVisibility(8);
        }
    }

    /* loaded from: classes2.dex */
    public class b implements Runnable {
        public b() {
        }

        @Override // java.lang.Runnable
        public void run() {
            VideoActivity.this.finish();
        }
    }

    @Override // android.app.Activity
    public void onBackPressed() {
        setRequestedOrientation(1);
        new Handler().postDelayed(new b(), 200L);
    }

    @Override // android.app.Activity
    public void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        requestWindowFeature(1);
        getWindow().setFlags(1024, 1024);
        getWindow().addFlags(128);
        setRequestedOrientation(0);
        getWindow().getDecorView().setSystemUiVisibility(4098);
        setContentView(R.layout.activity_video);
        this.f5701b = (VideoView) findViewById(R.id.videoplayer);
        this.f5705f = (ProgressBar) findViewById(R.id.videoProgress);
        MediaController mediaController = new MediaController(this);
        mediaController.setAnchorView(this.f5701b);
        Bundle extras = getIntent().getExtras();
        if (extras != null) {
            if (extras.containsKey("videoUrl")) {
                this.f5702c = extras.getString("videoUrl");
            }
            if (extras.containsKey("loop")) {
                this.f5703d = extras.getBoolean("loop", false);
            }
            if (extras.containsKey("currenttime")) {
                this.f5704e = extras.getInt("currenttime");
            }
        }
        Uri parse = Uri.parse(this.f5702c);
        this.f5701b.setMediaController(mediaController);
        this.f5701b.setVideoURI(parse);
        this.f5701b.setOnPreparedListener(new a());
    }
}