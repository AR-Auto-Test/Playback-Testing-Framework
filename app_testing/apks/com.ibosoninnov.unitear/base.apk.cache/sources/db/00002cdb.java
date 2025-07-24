package com.ibosoninnov.unitear.activities;

import android.annotation.SuppressLint;
import android.content.Context;
import android.os.Bundle;
import android.view.MotionEvent;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import androidx.cardview.widget.CardView;
import b.b.c.h;
import b.j.c.a;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.ibosoninnov.unitear.R;
import com.ibosoninnov.unitear.activities.Help2Activity;
import java.util.Objects;
import pl.droidsonroids.gif.GifImageView;

/* loaded from: classes2.dex */
public class Help2Activity extends h {
    public ImageView r;
    public GifImageView s;
    public Button t;
    public int u = 0;
    public LinearLayout v;
    public ScrollView w;
    public TextView x;
    public float y;
    public float z;

    @Override // b.b.c.h, b.q.b.d, androidx.activity.ComponentActivity, b.j.b.e, android.app.Activity
    @SuppressLint({"ClickableViewAccessibility"})
    public void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        setContentView(R.layout.activity_help2);
        this.r = (ImageView) findViewById(R.id.backBtn);
        this.w = (ScrollView) findViewById(R.id.helpFrameLayout);
        this.t = (Button) findViewById(R.id.nextBtn);
        this.v = (LinearLayout) findViewById(R.id.linearLayout);
        this.s = (GifImageView) findViewById(R.id.img);
        this.x = (TextView) findViewById(R.id.content);
        Context applicationContext = getApplicationContext();
        Object obj = a.f2074a;
        ((CardView) this.v.getChildAt(0)).setCardBackgroundColor(applicationContext.getColor(R.color.newBtn));
        this.x.setText(getResources().getString(R.string.help2__title1));
        this.s.setImageResource(R.drawable.movement);
        this.r.setOnClickListener(new View.OnClickListener() { // from class: c.e.b.df.c
            @Override // android.view.View.OnClickListener
            public final void onClick(View view) {
                Help2Activity.this.finish();
            }
        });
        this.w.setOnTouchListener(new View.OnTouchListener() { // from class: c.e.b.df.b
            @Override // android.view.View.OnTouchListener
            public final boolean onTouch(View view, MotionEvent motionEvent) {
                Help2Activity help2Activity = Help2Activity.this;
                Objects.requireNonNull(help2Activity);
                int action = motionEvent.getAction();
                if (action == 0) {
                    help2Activity.y = motionEvent.getX();
                } else if (action == 1) {
                    float x = motionEvent.getX();
                    help2Activity.z = x;
                    float f2 = help2Activity.y - x;
                    if (Math.abs(f2) > StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                        if (f2 >= StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                            int i = help2Activity.u;
                            if (i < 3) {
                                int i2 = i + 1;
                                help2Activity.u = i2;
                                help2Activity.v(i2);
                            }
                        } else {
                            int i3 = help2Activity.u;
                            if (i3 > 0) {
                                int i4 = i3 - 1;
                                help2Activity.u = i4;
                                help2Activity.v(i4);
                            }
                        }
                    }
                    for (int i5 = 0; i5 < help2Activity.v.getChildCount(); i5++) {
                        CardView cardView = (CardView) help2Activity.v.getChildAt(i5);
                        if (i5 == help2Activity.u) {
                            Context applicationContext2 = help2Activity.getApplicationContext();
                            Object obj2 = b.j.c.a.f2074a;
                            cardView.setCardBackgroundColor(applicationContext2.getColor(R.color.newBtn));
                        } else {
                            Context applicationContext3 = help2Activity.getApplicationContext();
                            Object obj3 = b.j.c.a.f2074a;
                            cardView.setCardBackgroundColor(applicationContext3.getColor(R.color.gray11));
                        }
                    }
                }
                return false;
            }
        });
        this.t.setOnClickListener(new View.OnClickListener() { // from class: c.e.b.df.a
            @Override // android.view.View.OnClickListener
            public final void onClick(View view) {
                Help2Activity help2Activity = Help2Activity.this;
                int i = help2Activity.u + 1;
                help2Activity.u = i;
                help2Activity.v(i);
                for (int i2 = 0; i2 < help2Activity.v.getChildCount(); i2++) {
                    CardView cardView = (CardView) help2Activity.v.getChildAt(i2);
                    if (i2 == help2Activity.u) {
                        Context applicationContext2 = help2Activity.getApplicationContext();
                        Object obj2 = b.j.c.a.f2074a;
                        cardView.setCardBackgroundColor(applicationContext2.getColor(R.color.newBtn));
                    } else {
                        Context applicationContext3 = help2Activity.getApplicationContext();
                        Object obj3 = b.j.c.a.f2074a;
                        cardView.setCardBackgroundColor(applicationContext3.getColor(R.color.gray11));
                    }
                }
            }
        });
    }

    public final void v(int i) {
        if (i == 3) {
            finish();
            return;
        }
        if (i == 0) {
            this.x.setText(getResources().getString(R.string.help2__title1));
            this.s.setImageResource(R.drawable.movement);
        } else if (i == 1) {
            this.x.setText(getResources().getString(R.string.help2__title2));
            this.s.setImageResource(R.drawable.scale);
        } else if (i == 2) {
            this.x.setText(getResources().getString(R.string.help2__title3));
            this.s.setImageResource(R.drawable.rotate);
        }
        if (i == 2) {
            this.t.setText(getResources().getString(R.string.finish));
        } else {
            this.t.setText(getResources().getString(R.string.next));
        }
    }
}