import React from 'react';
import { Input } from "./components/ui/input";
import { Textarea } from "./components/ui/textarea";
import { Button } from "./components/ui/button";
import { Card, CardContent } from "./components/ui/card";
import { Particles } from "./components/magicui/particles";

const Contact = () => {
  return (
    <div className="relative flex w-full items-center justify-center min-h-screen bg-background px-4 py-12">
      <div className="relative z-10 w-full max-w-2xl">
        <Card className="w-full shadow-2xl border border-border">
          <CardContent className="p-8 sm:p-10">
            <h2 className="text-3xl font-bold mb-8 text-foreground text-center">Contact Us</h2>
            <form className="space-y-6">
              <div>
                <label htmlFor="name" className="block text-sm font-medium mb-1 text-foreground">
                  Name
                </label>
                <Input id="name" type="text" placeholder="Your Name" />
              </div>
              <div>
                <label htmlFor="email" className="block text-sm font-medium mb-1 text-foreground">
                  Email
                </label>
                <Input id="email" type="email" placeholder="you@example.com" />
              </div>
              <div>
                <label htmlFor="message" className="block text-sm font-medium mb-1 text-foreground">
                  Message
                </label>
                <Textarea id="message" placeholder="Your message..." rows={6} />
              </div>
              <Button type="submit" className="w-full">
                Send Message
              </Button>
            </form>
          </CardContent>
        </Card>
      </div>

      <Particles
        className="absolute inset-0 z-0"
        quantity={100}
        ease={80}
        refresh
      />
    </div>
  );
};

export default Contact;
