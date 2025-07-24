package com.ibosoninnov.unitear;

import android.os.Bundle;
import android.view.MenuItem;
import android.view.View;
import android.widget.TextView;
import b.b.c.h;
import b.b.c.u;
import java.util.ArrayList;
import java.util.Arrays;

/* loaded from: classes2.dex */
public class AboutTheApp extends h {
    public static final /* synthetic */ int r = 0;
    public TextView s;
    public ArrayList<String> t;

    public void onBack(View view) {
        finish();
    }

    @Override // b.b.c.h, b.q.b.d, androidx.activity.ComponentActivity, b.j.b.e, android.app.Activity
    public void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        setContentView(R.layout.activity_about_the_app);
        r().c(true);
        ((u) r()).f620g.setTitle(getResources().getString(R.string.about_the_app));
        this.t = new ArrayList<>(Arrays.asList(getResources().getString(R.string.how_to_use), getResources().getString(R.string.about_the_app), getResources().getString(R.string.help)));
        this.s = (TextView) findViewById(R.id.tvHeading);
        this.s.setText(this.t.get(getIntent().getExtras().getInt("position")));
    }

    @Override // android.app.Activity
    public boolean onOptionsItemSelected(MenuItem menuItem) {
        if (menuItem.getItemId() != 16908332) {
            return super.onOptionsItemSelected(menuItem);
        }
        finish();
        return true;
    }
}