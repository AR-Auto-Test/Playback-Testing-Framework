package com.ibosoninnov.unitear.activities;

import android.os.Bundle;
import android.view.View;
import android.widget.ImageView;
import androidx.cardview.widget.CardView;
import b.b.c.h;
import com.ibosoninnov.unitear.R;

/* loaded from: classes2.dex */
public class GuidanceActivity extends h {
    public ImageView r;

    /* loaded from: classes2.dex */
    public class a implements View.OnClickListener {
        public a() {
        }

        @Override // android.view.View.OnClickListener
        public void onClick(View view) {
            GuidanceActivity.this.finish();
        }
    }

    @Override // b.b.c.h, b.q.b.d, androidx.activity.ComponentActivity, b.j.b.e, android.app.Activity
    public void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        setContentView(R.layout.activity_guidance);
        CardView cardView = (CardView) findViewById(R.id.arGalleryButton);
        CardView cardView2 = (CardView) findViewById(R.id.arScannerButton);
        ImageView imageView = (ImageView) findViewById(R.id.backBtn);
        this.r = imageView;
        imageView.setOnClickListener(new a());
    }
}